class Department(object):
    name = ""

    def __init__(self, name: str):
        self.name = name


class Book(object):
    book_id = 0
    department = Department("")

    def __init__(self, id_: int, dep: Department):
        self.id = id_
        self.department = dep


class Person(object):
    first_name = ""
    last_name = ""
    book_list = []

    def __init__(self, first, last):
        self.first_name = first
        self.last_name = last

    def change_info(self, first, last):
        self.first_name = first
        self.last_name = last

    def check_out_book(self, book: Book):
        self.book_list.append(book)

    def return_book(self, book: Book):
        if book in self.book_list:
            index = self.book_list.index(book)
            self.book_list.pop(index)


class Student(Person):
    student_id = 0

    def __init__(self, id_: int, first, last):
        super(Student, self).__init__(self, first, last)
        self.student_id = id_


class Faculty(Person):
    faculty_id = 0
    department = Department("")

    def __init__(self, id_: int, dep: Department, first, last):
        super(Faculty, self).__init__(self, first, last)
        self.faculty_id = id_
        self.department = dep

