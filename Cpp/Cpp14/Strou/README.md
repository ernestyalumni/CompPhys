cf. [`std::array`](http://en.cppreference.com/w/cpp/container/array)

# `std::array`  

Defined in header `<array>`  

```  
template<
	class T,
	std::size_t N
> struct array;  
```  

"This container is an aggregate type with the same semantics as a struct holding a [C-style array](http://en.cppreference.com/w/cpp/language/array) `T[N]` as its only non-static data member.  

From pp. 208, Sec. 8.2.4 "Structures and Arrays" of Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup introduces the "API" or reference for standard library's `std::array` and gave this *partial*, simplified code to explain its structure/implementation by the actual standard library:  

```  
template<typename T, size_t N> 
struct array {	// simplified (see Sec. 34.2.1)  
	T elem[N]; 
	
	T* begin() noexcept { return elem; }
	const T* begin() const noexcept { return elem; }
	T* end() noexcept { return elem+N; }
	const T* end() const noexcept { return elem+N; }
	
	constexpr size_t size() noexcept; 
	
	T& operator[](size_t n) { return elem[n]; }
	const T& operator[](size_t n) const { return elem[n]; }
	
	T* data() noexcept { return elem; }
	const T* data() const noexcept { return elem; }
	
	// ... 
};  
```  
  
# `std::ostream`  

cf. [`std::ostream` in `cplusplus.com`](http://www.cplusplus.com/reference/ostream/ostream/)

`<ostream> <iostream>`  

Output stream objects can write sequences of characters and represent other kinds of data.  
