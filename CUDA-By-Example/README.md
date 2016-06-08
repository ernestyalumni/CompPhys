# `CUDA-By-Example` CUDA by Example
#### cf. Jason Sanders, Edward Kandrot. **CUDA by Example: An Introduction to General-Purpose GPU Programming**

I also cloned in this same repository the github repository from [jiekebo](https://github.com/jiekebo) for [CUDA-By-Example](https://github.com/jiekebo/CUDA-By-Example), and this is a general observation: it may help to first search on github for the code you seek because it seems likely that someone already wrote it.

## Note on examples out of CUDA by Example

It seems that a header file `book.h` out of the `common` subdirectory is needed to run the scripts.  I've tried to write my own scripts without needing the `book.h` header.  However, it's found in jiekebo's repository and in this repository, in the subdirectory (from here) `CUDA-By-Example`.  I don't know the rationale behind `book.h` or why the authors made you need it for their examples (as I am reading the book, it's not explained (!!!)).

## Dictionary between files on this github subdirectory to code in **CUDA By Example**, *Sanders and Kandrot*

I'm also looking at Bjarne Stroustrup's **A Tour of C++** (2013) Addison-Wesley, and Stroustrup (2013) refers to this text.  

| filename       |   pp.  | (Sub)Section             | Description                  |
| -------------- | :----: | :--------------------:   | :--------------------------: |
| helloworld.c   | 22     | 3.2 A First Program      | Hello world in C             |
| helloworld.cpp | 22     | 3.2 A First Program      | Hello world in C++; also in pp. 3, Section 1.3 Hello World! of Stroustrup (2013) |
| helloworld.cu  | 23     | 3.2.2 A Kernel Call      | Hello world in CUDA C        |
| [add-pass.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/add-pass.cu) |  25  | 3.2.3 Passing Parameters |
| [cpuvecsum.c](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/cpuvecsum.c) |  40  | 4.2.1 Summing Vectors    | Sum vectors as an C array    |
| [gpuvecsum.cu](https://github.com/ernestyalumni/CompPhys/blob/master/CUDA-By-Example/gpuvecsum.cu) | 41 | 4.2.1 Summing Vectors    | Sum arrays as vectors in GPU |
