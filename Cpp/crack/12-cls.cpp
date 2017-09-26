/**
  * @file  : 12-cls.cpp
  * @brief : Chapter 12  
  * @ref   : Gayle Laekmann McDowell.  Cracking the Coding Interview
  */  
#include <iostream>

#define NAME_SIZE 50 // Defines a macro

class Person {
    int id; // all members are private by default
    char name[NAME_SIZE];

    public:
        void aboutMe() {
            std::cout << "I am a person."; 
        }
};

class Student : public Person {
    public:
        void aboutMe() {
            std::cout << "I am a student.";
        }
};

int main() {
    Student * p = new Student();
    p->aboutMe(); // prints "I am a student."
    delete p; // Important!  Make sure to delete allocated memory.  
    return 0;
}

