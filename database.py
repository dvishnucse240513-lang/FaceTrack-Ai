import sqlite3
import pickle
import datetime
import threading
import io
import zipfile
from xml.sax.saxutils import escape

class Database:
    def __init__(self, db_name='attendance.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.lock = threading.RLock()
        self.create_tables()

    def create_tables(self):
        with self.lock:
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA synchronous=NORMAL')
            self.conn.execute('PRAGMA temp_store=MEMORY')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL
            )''')
            self._ensure_column('students', 'department', "TEXT NOT NULL DEFAULT ''")
            self.conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                marked_at TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )''')
            self._ensure_column('attendance', 'marked_at', 'TEXT')
            self._backfill_marked_at()
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_attendance_marked_at ON attendance(marked_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_students_name ON students(name)')
            self.conn.commit()

    def _ensure_column(self, table_name, column_name, column_definition):
        cursor = self.conn.execute(f'PRAGMA table_info({table_name})')
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            self.conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}')

    def _backfill_marked_at(self):
        cursor = self.conn.execute('SELECT id, date, time FROM attendance WHERE marked_at IS NULL OR marked_at = ""')
        rows = cursor.fetchall()
        for row_id, date_text, time_text in rows:
            marked_at = self._combine_date_time(date_text, time_text)
            self.conn.execute('UPDATE attendance SET marked_at = ? WHERE id = ?', (marked_at, row_id))

    def _combine_date_time(self, date_text, time_text):
        try:
            record_date = datetime.date.fromisoformat(date_text)
        except (TypeError, ValueError):
            record_date = datetime.date.today()

        for time_format in ('%I:%M:%S %p', '%H:%M:%S'):
            try:
                record_time = datetime.datetime.strptime(time_text, time_format).time()
                break
            except (TypeError, ValueError):
                record_time = datetime.time()
        else:
            record_time = datetime.time()

        return datetime.datetime.combine(record_date, record_time).replace(microsecond=0).isoformat()

    def _format_attendance_row(self, row):
        marked_at = self._parse_marked_at(row[4], row[2], row[3])
        return {
            'name': row[0],
            'department': row[1],
            'date': marked_at.date().isoformat(),
            'time': self._format_display_time(marked_at),
            'marked_at': marked_at.isoformat()
        }

    def _parse_marked_at(self, marked_at, date_text=None, time_text=None):
        try:
            return datetime.datetime.fromisoformat(marked_at)
        except (TypeError, ValueError):
            return datetime.datetime.fromisoformat(self._combine_date_time(date_text, time_text))

    def _format_display_time(self, value):
        return value.strftime('%I:%M:%S %p').lstrip('0')

    def add_student(self, name, department, embedding):
        """Add a new student with face embedding."""
        emb_data = pickle.dumps(embedding)
        with self.lock:
            cursor = self.conn.execute(
                'INSERT INTO students (name, department, embedding) VALUES (?, ?, ?)',
                (name, department, emb_data)
            )
            self.conn.commit()
            return cursor.lastrowid

    def get_students(self):
        """Retrieve all students with their embeddings."""
        with self.lock:
            cursor = self.conn.execute('SELECT id, name, department, embedding FROM students ORDER BY name')
            rows = cursor.fetchall()
        return [{'id': row[0], 'name': row[1], 'department': row[2], 'embedding': pickle.loads(row[3])} for row in rows]

    def mark_attendance(self, student_id):
        """Mark attendance for a student if not already marked today."""
        now = datetime.datetime.now().replace(microsecond=0)
        today = now.date().isoformat()
        display_time = self._format_display_time(now)
        with self.lock:
            cursor = self.conn.execute('SELECT id FROM attendance WHERE student_id = ? AND date = ?', (student_id, today))
            if cursor.fetchone():
                return False
            self.conn.execute(
                'INSERT INTO attendance (student_id, date, time, marked_at) VALUES (?, ?, ?, ?)',
                (student_id, today, display_time, now.isoformat())
            )
            self.conn.commit()
            return True

    def get_marked_student_ids_today(self):
        """Get student ids that already have attendance today."""
        today = datetime.date.today().isoformat()
        with self.lock:
            cursor = self.conn.execute('SELECT student_id FROM attendance WHERE date = ?', (today,))
            return {row[0] for row in cursor.fetchall()}

    def get_attendance_today(self):
        """Get today's attendance records."""
        today = datetime.date.today().isoformat()
        with self.lock:
            cursor = self.conn.execute('''SELECT s.name, s.department, a.date, a.time, a.marked_at FROM attendance a
                                          JOIN students s ON a.student_id = s.id
                                          WHERE a.date = ? ORDER BY a.marked_at''', (today,))
            return self._format_attendance_rows(cursor.fetchall())

    def get_attendance_count_today(self):
        """Get today's attendance count."""
        today = datetime.date.today().isoformat()
        with self.lock:
            cursor = self.conn.execute('SELECT COUNT(*) FROM attendance WHERE date = ?', (today,))
            return cursor.fetchone()[0]

    def get_all_attendance(self, limit=None):
        """Get all attendance records."""
        query = '''SELECT s.name, s.department, a.date, a.time, a.marked_at FROM attendance a
                   JOIN students s ON a.student_id = s.id
                   ORDER BY a.marked_at DESC, a.date DESC, a.time DESC'''
        params = ()
        if limit is not None:
            query += ' LIMIT ?'
            params = (int(limit),)
        with self.lock:
            cursor = self.conn.execute(query, params)
            return self._format_attendance_rows(cursor.fetchall())

    def get_student_summaries(self):
        """Get student list with attendance counts."""
        with self.lock:
            cursor = self.conn.execute('''SELECT s.id, s.name, s.department, COUNT(a.id) as attendance_count
                                          FROM students s
                                          LEFT JOIN attendance a ON s.id = a.student_id
                                          GROUP BY s.id
                                          ORDER BY s.name''')
            return [{'id': row[0], 'name': row[1], 'department': row[2], 'attendance_count': row[3]} for row in cursor.fetchall()]

    def get_student(self, student_id):
        """Get a single student record."""
        with self.lock:
            cursor = self.conn.execute('SELECT id, name, department FROM students WHERE id = ?', (student_id,))
            row = cursor.fetchone()
        return {'id': row[0], 'name': row[1], 'department': row[2]} if row else None

    def get_student_attendance(self, student_id):
        """Get attendance history for one student."""
        with self.lock:
            cursor = self.conn.execute('''SELECT date, time, marked_at FROM attendance
                                          WHERE student_id = ?
                                          ORDER BY marked_at DESC, date DESC, time DESC''', (student_id,))
            rows = cursor.fetchall()
            return [
                {
                    'date': marked_at.date().isoformat(),
                    'time': self._format_display_time(marked_at),
                    'marked_at': marked_at.isoformat()
                }
                for marked_at in (self._parse_marked_at(row[2], row[0], row[1]) for row in rows)
            ]

    def _format_attendance_rows(self, rows):
        return [self._format_attendance_row(row) for row in rows]

    def delete_student(self, student_id):
        """Delete a student and all of their attendance records."""
        with self.lock:
            cursor = self.conn.execute('SELECT id FROM students WHERE id = ?', (student_id,))
            if not cursor.fetchone():
                return False
            self.conn.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
            self.conn.execute('DELETE FROM students WHERE id = ?', (student_id,))
            self.conn.commit()
            return True

    def get_total_students(self):
        """Get total number of students."""
        with self.lock:
            cursor = self.conn.execute('SELECT COUNT(*) FROM students')
            return cursor.fetchone()[0]

    def export_csv(self):
        """Export all attendance to CSV string."""
        import csv

        def excel_text(value):
            value = '' if value is None else str(value)
            return f'="{value.replace(chr(34), chr(34) + chr(34))}"'

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Name', 'Department', 'Date', 'Time'])
        for row in self.get_all_attendance():
            writer.writerow([row['name'], row['department'], row['date'], excel_text(row['time'])])
        return output.getvalue()

    def export_xlsx(self):
        """Export all attendance to an Excel workbook."""
        rows = self.get_all_attendance()
        output = io.BytesIO()

        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as workbook:
            workbook.writestr('[Content_Types].xml', self._xlsx_content_types())
            workbook.writestr('_rels/.rels', self._xlsx_root_rels())
            workbook.writestr('xl/workbook.xml', self._xlsx_workbook())
            workbook.writestr('xl/_rels/workbook.xml.rels', self._xlsx_workbook_rels())
            workbook.writestr('xl/styles.xml', self._xlsx_styles())
            workbook.writestr('xl/worksheets/sheet1.xml', self._xlsx_sheet(rows))

        output.seek(0)
        return output

    def _xlsx_sheet(self, rows):
        sheet_rows = [self._xlsx_header_row()]
        for index, row in enumerate(rows, start=2):
            marked_at = datetime.datetime.fromisoformat(row['marked_at'])
            sheet_rows.append(
                '<row r="{0}">'
                '{1}{2}{3}{4}'
                '</row>'.format(
                    index,
                    self._xlsx_text_cell('A', index, row['name']),
                    self._xlsx_text_cell('B', index, row['department']),
                    self._xlsx_number_cell('C', index, self._excel_date_serial(marked_at.date()), 2),
                    self._xlsx_number_cell('D', index, self._excel_time_serial(marked_at.time()), 3)
                )
            )

        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetViews><sheetView workbookViewId="0"/></sheetViews>
    <sheetFormatPr defaultRowHeight="15"/>
    <cols>
        <col min="1" max="1" width="24" customWidth="1"/>
        <col min="2" max="2" width="18" customWidth="1"/>
        <col min="3" max="3" width="14" customWidth="1"/>
        <col min="4" max="4" width="16" customWidth="1"/>
    </cols>
    <sheetData>{rows}</sheetData>
</worksheet>'''.format(rows=''.join(sheet_rows))

    def _xlsx_header_row(self):
        headings = ['Name', 'Department', 'Date', 'Time']
        cells = ''.join(self._xlsx_text_cell(chr(65 + index), 1, heading, 1) for index, heading in enumerate(headings))
        return f'<row r="1">{cells}</row>'

    def _xlsx_text_cell(self, column, row, value, style=0):
        style_attr = f' s="{style}"' if style else ''
        text = escape('' if value is None else str(value))
        return f'<c r="{column}{row}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'

    def _xlsx_number_cell(self, column, row, value, style):
        return f'<c r="{column}{row}" s="{style}"><v>{value}</v></c>'

    def _excel_date_serial(self, value):
        return (value - datetime.date(1899, 12, 30)).days

    def _excel_time_serial(self, value):
        seconds = value.hour * 3600 + value.minute * 60 + value.second
        return round(seconds / 86400, 10)

    def _xlsx_content_types(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    def _xlsx_root_rels(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    def _xlsx_workbook(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets><sheet name="Attendance" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''

    def _xlsx_workbook_rels(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    def _xlsx_styles(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <numFmts count="2">
        <numFmt numFmtId="164" formatCode="yyyy-mm-dd"/>
        <numFmt numFmtId="165" formatCode="h:mm:ss AM/PM"/>
    </numFmts>
    <fonts count="2">
        <font><sz val="11"/><name val="Calibri"/></font>
        <font><b/><sz val="11"/><name val="Calibri"/></font>
    </fonts>
    <fills count="2">
        <fill><patternFill patternType="none"/></fill>
        <fill><patternFill patternType="gray125"/></fill>
    </fills>
    <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
    <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
    <cellXfs count="4">
        <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
        <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
        <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
        <xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    </cellXfs>
    <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
