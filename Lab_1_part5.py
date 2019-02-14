# Class 1
class Department(object):
    def __init__(self, name: str): # INIT Constructor
        self.name = name


# Class 2
class Book(object):
    # used for auto creating unique ids when book is created
    __numBooks =0   # private variable

    def __init__(self, name: str, dep: Department): # INIT Constructor
        self.name = name
        self.department = dep
        self.id = Book.__numBooks + 1
        Book.__numBooks += 1
        # keeps track of which student or staff has book checked
        self.owner = 0


# Class 3
class Person(object):
    __num_persons = 0

    def __init__(self, first, last): # INIT Constructor
        self.first_name = first
        self.last_name = last
        self.book_list = []
        self.id = Person.__num_persons + 1
        Person.__num_persons += 1

    def change_info(self, first, last):
        self.first_name = first
        self.last_name = last

    def check_out_book(self, book: Book):
        if book.owner == 0:
            self.book_list.append(book)
            book.owner = self.id
        else:
            print("Book already checked out")

    def return_book(self, book: Book):
        if book in self.book_list:
            index = self.book_list.index(book)
            self.book_list.pop(index)
            book.owner = 0


# Class 4
class Student(Person):

    def __init__(self, first, last): # INIT Constructor
        super(Student, self).__init__(first, last)  # Inheritance Super call



# Class 5
class Faculty(Person):

    def __init__(self, dep: Department, first, last):  # INIT Constructor
        super(Faculty, self).__init__(first, last)  # Inheritance Super Call
        self.department = dep




# create departments
English = Department("English")
Spanish = Department("Spanish")

# create books
book_1 = Book("1st Grade Spelling", English)
book_2 = Book("Learn Spanish!", Spanish)
book_3 = Book("Conversational Spanish", Spanish)
book_4 = Book("Shakespeare", English)
# creates library
library=[book_1,book_2,book_3,book_4]

# Create Students
Student_1 = Student("Billy", "Ray")
Student_2 = Student("Sally", "Sue")
Student_3 = Student("Sarah", "Conely")
# creates class list
Students = [Student_1, Student_2, Student_3]

# Create Faculty
faculty_1 = Faculty(English, "Kim", "Kardashian")
faculty_2 = Faculty(Spanish, "Bob", "Cortez")
# creates Faculty list
Staff = [faculty_1, faculty_2]


# demonstrates constructors for persons worked correctly
for student in Students:
    print("student id: {0} First: {1} Last: {2} Books Out: {3}".format(student.id, student.first_name,
                                                                       student.last_name, student.book_list))
for faculty in Staff:
    print("staff id: {0} First: {1} Last: {2} Books Out: {3}".format(faculty.id, faculty.first_name, faculty.last_name,
                                                                     faculty.book_list))

# Demonstriates that Book Constructors work correctly

for book in library:
    print("Book ID: {0} name: {1} Department: {2} Owner: {3}".format(book.id, book.name, book.department.name,
                                                                     book.owner))

faculty_2.check_out_book(book_1)
Student_1.check_out_book(book_1)
Student_1.check_out_book(book_2)
Student_1.check_out_book(book_3)


for student in Students:
    print("student id: {0} First: {1} Last: {2} Books Out: {3}".format(student.id, student.first_name,
                                                                       student.last_name, student.book_list))
    for book in student.book_list:
        print(book.name)

for faculty in Staff:
    print("staff id: {0} First: {1} Last: {2} Books Out: {3}".format(faculty.id, faculty.first_name, faculty.last_name,
                                                                     faculty.book_list))
    for book in faculty.book_list:
        print(book.name)


Student_1.return_book(book_2)


for student in Students:
    print("student id: {0} First: {1} Last: {2} Books Out: {3}".format(student.id, student.first_name,
                                                                       student.last_name, student.book_list))
    for book in student.book_list:
        print(book.name)