# OpenCV - C++ examples for OpenCV, Open Computer Vision

## Abstract

Here is a collection of code snippets, pedagogical examples, code illustrating usage or using OpenCV, Open Computer Vision.

## Table of Contents  
Be aware that this Table of Contents maybe incomplete (one way around this is to search for key words with some kind of search function).  

- Installing OpenCV (on Fedora Linux): my experience

### Code Listing

| codename        | directory      | Chapter | Section | page (pp) | Description            |
| --------------- | -------------- | :-----: | ------- | --------- | ---------------------- |
| argcargv.cpp    | ./             |         |         |           | This is my own pedagogical example of using argc, argv in C++; it's also a good example of looping through an array as a pointer |
| program1.cpp    | ./progs/ch02/  | 2       | 2.1.1   | 10        | Scientific Hello World!|
| program7.cpp    | ./progs/ch02/  | 2       | 2.5.2   | 34        | Pointers and arrays in C++ |

## Installing OpenCV (on Fedora Linux): my experience

The following 2 commands run in the terminal prompt *in an administrator account* worked for me:

```sudo dnf install opencv

sudo dnf install opencv-devel  ```

I needed `opencv-devel` because, as explained here, [How to install OpenCV for C](https://ask.fedoraproject.org/en/question/89972/how-to-install-opencv-for-c/).

