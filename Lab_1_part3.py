
class Department(object):
    name = ""

class Book (object):
    book_id = 0
    department = ""

class Person (object):
    first_name =""
    last_name =""
    book_list =[]

    def __init__(self ,first, last):
        self.first_name = first
        self.last_name = last

    def change_info (self, first, last):
        self.first_name = first
        self.last_name = last

    def check_out_book (self , book: Book):
        self.book_list.append(Book)

    def return_book (self, book: Book):
        if book in self.book_list:
            index = self.book_list.index(book)
            self.book_list.pop(index)


class Student (Person):
    student_id = 0

class Faculty (Person):
    faculty_id = 0
    department = ""




