# OpenCV - C++ examples for OpenCV, Open Computer Vision

## Abstract

Here is a collection of code snippets, pedagogical examples, code illustrating usage or using OpenCV, Open Computer Vision.

## Table of Contents  
Be aware that this Table of Contents maybe incomplete (one way around this is to search for key words with some kind of search function).  

- Installing OpenCV (on Fedora Linux): my experience

### Code Listing

| codename         | directory         | Description            | URL reference link     |
| ---------------- | :---------------- | :--------------------: | ---------------------- |
| DisplayImage.cpp | `./DisplayImage/` | Simple program to display an image; OpenCV's "Hello World"; also you have to make a `CMakeFile` | [Using OpenCV with gcc and CMake](http://docs.opencv.org/3.1.0/db/df5/tutorial_linux_gcc_cmake.html)  | 


## Installing OpenCV (on Fedora Linux): my experience

The following 2 commands run in the terminal prompt *in an administrator account* worked for me:

```
sudo dnf install opencv  

sudo dnf install opencv-devel  
```

I needed `opencv-devel` because, as explained here, [How to install OpenCV for C](https://ask.fedoraproject.org/en/question/89972/how-to-install-opencv-for-c/).

Then I checked that `opencv` and `opencv2` was in my `/usr/install` so that it can be included in my code/programs/scripts by `cd`'ing into that directory and manually looking it up.  