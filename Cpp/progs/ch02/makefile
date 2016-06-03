/* 
 * cf. M. Hjorth-Jensen, Computational Physics, University of Oslo (2015) 
 * http://www.mn.uio.no/fysikk/english/people/aca/mhjensen/ 
 * Ernest Yeung (I typed it up and made modifications)
 * ernestyalumni@gmail.com
 * MIT License
 * cf. Chapter 2 Introduction to C++ and Fortran 
 * 2.1.1 Scientific hello world
 */
/*
	To make this into a makefile, rename this file into makefile and then running
	make
	on the terminal prompt should make the file
*/

/* 
   ** ORIGINAL makefile template **

 General makefile for c - choose PROG =   name of given program

 Here we define compile option, libraries and the target
CC= c++ -Wall
PROG= myprogram

 Here we make the executable file
${PROG} :          ${PROG}.o
                   ${CC} ${PROG}.o -o ${PROG}

 whereas here we create the object file

${PROG}.o :        ${PROG}.cpp
	           ${CC} -c ${PROG}.cpp

*/

CC= g++ -Wall
PROG= program1

# Here we make the executable file
${PROG} : ${PROG}.o 
	${CC} ${PROG}.o -o ${PROG}

# whereas here we create the object file

${PROG}.o : ${PROG}.cpp
	${CC} -c ${PROG}.cpp


