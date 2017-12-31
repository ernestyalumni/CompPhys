# I/O Streams, File I/O  

cf. Ch. 38 I/O Streams; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed. 2013  

in `<istream> <iostream>`  

`ostream` converts typed objects -> stream of characters (bytes)  
`istream` converts stream of characters (bytes) to typed objects  

cf. Sec. 38.2 The I/O Stream Hierarchy, Ch. 38 I/O Streams; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   
  
`istream` can be connected to an input device (e.g. keyboard), file, or `string`  
`ostream` can be connected to an output device (e.g. text window, or HTML engine), file, or `string`  


![`File:std-basic fstream-inheritance.svg`](http://upload.cppreference.com/mwiki/images/f/f1/std-basic_fstream-inheritance.svg)  

cf. Sec. 38.2.1 File Streams, Ch. 38 I/O Streams; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed. 2013   

in `<fstream>`  
* `ifstream`s for reading from file  
* `ofstream`s for writing from file 
* `fstream`s for reading from and writing to file  

`fstream` (common pattern), 1st. from Stroustrup (2013):  

```  
template<typename C, typename Tr=char_traits<C>>  
class basic_fstream : public basic_iostream<C,Tr> {
	public:
		using char_type = C;
		using int_type = typename Tr::int_type;
		using pos_type = typename Tr::pos_type;		// for positions in file
		using off_type = typename Tr::off_type;		// for offsets in file 
		using traits_type = Tr; 
		// ... 
};  
```  

On the other hand, from [`cppreference.com`, Standard library header `<fstream>`](http://en.cppreference.com/w/cpp/header/fstream)  

class template `basic_fstream` - implements high-level file stream input/output, with typedef  
`fstream` - `basic_fstream<char>`  

Class `std::basic_fstream`  
```  
template <class charT, class traits=<char_traits<charT> >  
class basic_fstream : public basic_iostream<charT, traits> { 
	public:
		typedef charT char_type; 
		typedef typename traits::int_type int_type; 
		typedef typename traits::pos pos_type;
		typedef typename traiots::off_type off_type;
		typedef traits traits_type; 
		
		// constructors/destructor
		basic_fstream();
		explicit basic_fstream(const char* s,
								ios_base::openmode mode = ios_base::in|ios_base::out); 
		explicit basic_fstream(const string& s,
								ios_base::openmode mode = ios_base::in|ios_base::out); 
		basic_fstream(const basic_fstream& rhs) = delete;  // copy constructor 
		basic_fstream(basic_fstream&& rhs); // move constructor
		
		// Assign/swap:
		basic_fstream& operator=(const basic_fstream& rhs) = delete; 
		basic_fstream& operator=(basic_fstream&& rhs); 
		void swap(basic_fstream& rhs); 
		
		// Members:
		basic_filebuf<charT, traits>* rdbuf() const; 
		bool is_open() const;
		void open(const char* s, 
					ios_base::openmode mode = ios_base::in|ios_base::out); 
		void open(const string& s, 
					ios_base::openmode mode = ios_base::in|ios_base::out);
		void close(); 
	private:
		basic_filebuf<charT,traits> sb; 	// exposition only 
};
								
```  

cf. [`cppreference.com`, `std::basic_fstream`](http://en.cppreference.com/w/cpp/io/basic_fstream)  

### **Member functions** of `std::basic_fstream`, and thus `fstream`, which is, by definition `basic_fstream<char>`, in header `<fstream>`:  

`operator=` - moves the file stream, public member function  
`swap` - swaps 2 file streams, public member function  
`rdbuf` - returns underlying raw file device object, public member function  

#### File operations  
`is_open` - checks if stream has associated file, public member function  
`open` - opens file and associates it with the stream, public member function  
`close` - closes associated file, public member function  

### Inherited from `std::basic_istream`  

#### Unformatted input  

[`read`](http://en.cppreference.com/w/cpp/io/basic_istream/read) - extracts blocks of characters, public member function of `std::basic_istream`  



### Inherited from `std::basic_ostream`   

#### Member functions  

##### Unformatted input 

[`write`](http://en.cppreference.com/w/cpp/io/basic_ostream/write) - inserts blocks of characters, public member function of `std::basic_ostream`  

##### Positioning  

`tellp` - returns output position indicator, public member function of `std::basic_ostream`  
[`seekp`](http://en.cppreference.com/w/cpp/io/basic_ostream/seekp) - sets output position indicator, public member function of `std::basic_ostream`  


Indeed, also given in cf. pp. 1077 Sec. 38.2.1 File Streams, Ch. 38 I/O Streams; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed. 2013,  
set of `fstream` operations  

`basic_fstream<C,Tr>`  
`fstream fs {};`  - `fs` is file stream not attached to a file 
`fstream fs {s,m}` - `fs` is file stream opened for file called `s` with mode `m`; `s` can be `string` or C-style string  
`fstream fs {fs2};`  - move constructor `fs2` is moved to `fs`; `fs2` becomes unattached  
`fs=move(fs2)` - move assignment: `fs2` moved to `fs`; `fs2` becomes unattached  
`fs.swap(fs2)` - exchange states of `fs` and `fs2`  
`p=fs.rdbuf()` - `p` is ptr to `fs`'s file stream buffer (`basic_filebuf<C,Tr>`)  
`fs.is_open()` - Is `fs` open?  
`fs.open(s,m)` - open file called `s` with mode `m` and have `fs` refer to it; sets `fs`'s `failbit` if it couldn't open the file; `s` can be `string` or C-style string  
`fs.close()` - close file associated with `fs`  

string streams override `basic_ios` protected virtual functions `underflow()`, `pback-fail()`, `overflow()`, `setbuf()`, `seekoff()`, `seekpos()` (Sec. 38.6)  

file stream *doesn't have copy operations.*  
If you want 2 names to refer to same file stream, use a reference or a pointer, or carefully manipulate file `streambuf`s (Sec. 38.6)  

If an `fstream` fails to open, stream is in `bad()` state (Sec.38.3)  

#### Typedefs for std library header `<fstream>`  

```   
using ifstream = basic_ifstream<char>   ;
using ofstream = basic_ofstream<char> ;
using wofstream = basic_ofstream<wchar_t>;    
using fstream = basic_fstream<char> ; 
using wfstream = basic_fstream<wchar_t>;  
```  
     
You can open a file in 1 of several modes, as specified in `ios_base` (Sec. 38.4.4): 

`ios_base::app` - append (i.e. add to end of the file)  
`ios_base::ate` - "At end" (open and seek to the end)  
`ios_base::binary` - binary mode; beware of system-specific behavior  
`ios_base::in` 	- for reading 
`ios_base::out` - for writing 
`ios_base::truc` - truncate file to 0 length  

cf. Stream Modes Sec. iso.27.5.3.1.4, pp. 1077 Sec. 38.2.1 of Stroustrup (2013)  

In each case, exact effect of opening file may depend on operating system, and if operating system can't honor a request to open a file in a certain way, result will be a stream that's in `bad()` state (Sec. 38.3)  

cf. `./fstream_eg.cpp`

## String Streams  

cf. Sec. 38.2.2 String Streams, Ch. 38 I/O Streams; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed. 2013   

in `<sstream>`, std library provides streams to and from a `string`  
* `istringstream`s for reading from a `string`  
* `ostringstream`s for writing to `string`  
* `stringstream`s for reading from and writing to `string`  

common pattern of `stringstream` (cf. Stroustrup (2013), pp. 1078)  

```  
template<typename C, typename Tr = char traits<C>, typename A = allocator<C>>  
class basic_stringstream  
	: public basic_iostream<C,Tr> {
public:  
	using char_type = C;
	using int_type = typename Tr::int_type;
	using pos_type = typename Tr::pos_type;		// for positions in string  
	using off_type = typename Tr::off_type; 	// for offsets in string  
	using traits_type = Tr;
	using allocator_type = A;  
	
	// ... 
	
```   

From [`<sstream>` Std library header `cppreference.com`](http://en.cppreference.com/w/cpp/header/sstream)	

`stringstream` is typedef for `basic_stringstream<char>`

### Class `std::basic_stringstream`  

```  
template <class charT, 
			class traits = char_traits<charT>  
			class Allocator = allocator<charT>  
class basic_stringstream : public basic_iostream<charT,traits> {
	public:  
		// types:
		typedef charT char_type;  
		typedef typename traits::int_type int_type;  
		typedef typename traits::pos_type pos_type;  
		typedef typename traits::off_type off_type;  
		typedef traits traits_def;  
		typedef Allocator allocator_type;  
		
		// constructors/destructor  
		explicit basic_stringstream(ios_base::openmode which = ios_base::out|ios_base::in); 
		explicit basic_stringstream(const basic_string<charT,traits,Allocator>& str, 
									ios_base::openmode which = ios_base::out|ios_base::in); 
		basic_stringstream(const basic_stringstream& rhs) = delete;  
		basic_stringstream(basic_stringstream&& rhs);  							
									
		// ...	

```  

`stringstream` operations:  

`basic_stringstream<C,Tr,A>` (cf. Sec. iso27.8, Stroustrup (2013))  
`stringstream ss {m};` - `ss` is an empty string stream with mode `m`  
`stringstream ss {};`  - default constructor: `stringstream ss {ios_base::out|ios_base::in};`  
`stringstream ss {s,m};` - `ss` is string stream with its buffer initialized from `string s` with mode `m`  
`stringstream ss {s};` - `stringstream ss {s,ios_base::out|ios_base::in};   
`stringstream ss {ss2};` - Move constructor: `ss2` is moved to `ss`; `ss2` becomes empty  
`ss=move(ss2)` - Move assignment: `ss2` is moved to `ss`; `ss2` becomes empty  
`p=ss.rdbuf()` - `p` points to `ss`'s string stream buffer (a `basic_stringbuf<C,Tr,A>`)  
`s=ss.str()` - `s` is a `string` copy of the characters in `ss`: `s=ss.rdbuf()->str()`  
`ss.str(s)` - `ss`'s buffer is initialized from `string s: ss.rdbuf()->str(s)`; if `ss`'s mode is `ios::ate` ("at end"), values written to `ss` are added after the characters from `s`; otherwise values written overwrites characters from `s`  
`ss.swap(ss2)` - exchange states of `ss` and `ss2`  

Also,  cf. [`std::basic_istringstream` of `cppreference.com`](http://en.cppreference.com/w/cpp/io/basic_istringstream)

### Inherited from `std::basic_istream`  

#### Member functions  

##### Unformatted input  

`getline` - extracts characters until given character is found, public member function of `std::basic_istream`  

```  
basic_istream& getline( char_type* s, std::streamsize count);  
basic_istream& getline( char_type* s, std::streamsize count, char_type delim);  
```  
Example in `./getline_eg.cpp`  
cf. [`std::basic_istream::getline`](http://en.cppreference.com/w/cpp/io/basic_istream/getline)


default mode open modes for `istringstream` is `ios_base::in` and for `ostringstream`, default mode is `ios_base::out`  

string streams override `basic_ios` protected virtual functions `underflow()`, `pbackfail()`, `overflow()`, `setbuf()`, `seekoff()`, and `seekpos()` (Sec.38.6)  

string stream doesn't have copy operations; if you want 2 names to refer to same string stream, use reference or ptr.  

string stream aliases defined in `<sstream>`:  

```  
using istringstream = basic_istringstream<char>;  
using wistringstream = basic_istringstream<wchar_t>;  
using ostringstream = basic_ostringstream<char>;  
using wostringstream = basic_ostringstream<wchar_t>;  
using stringstream = basic_stringstream<char>;  
using wstringstream = basic_stringstream<wchar_t>;  
```  
















