/** functors.cu
 * 
 * \file functors.cu
 * \author Ernest Yeung
 * typed up by Ernest Yeung  ernestyalumni@gmail.com
 * \date 20161231
 * cf.  http://www.bogotobogo.com/cplusplus/functors.php
 * 
 * */
/* Compilation note: 
 * g++ -std=c++14 madesimple.cu -o madesimple.exe
 * */

#include <iostream> // std::cout

#include <vector>   // std::vector
#include <algorithm>  

#include <iterator>

/*
 * function-call operator must be declared as a member function
 * 
 */	
struct absValue
{
	float operator()(float f) {
		return f > 0 ? f: -f;
	}
};


/* 
 * example simulating a line
 * class object working as a functor taking x for given y-intercept (b) and
 * slope (a) giving us the correpsonding y coordinate
 * 
 * */

class Line {
	double a;			// slope
	double b; 			// y-intercept
	
	public:
		Line(double slope = 1, double yintercept = 1) : 
			a(slope), b(yintercept) { } 
		double operator()(double x){
			return a*x + b;
		}
};

// Here is a function object example
class Print { 
	public:
		void operator()(int elem) const { 
			std::cout << elem << " " ;
		}
};

// Function object using nontype template parameter

/* 
 * set template parameter with an integer value as non-type
 * 
 * set value we want by calling setValue()
 * 
 * */
template <int val>
void setValue(int& elem)
{
	elem = val;
};

template <typename T>
class PrintElements
{
	public:
		void operator()(T& elm) const { std::cout << elm << ' '; }
};

/* 
 * similar example, but this time it's adding some value to each element of the 
 * vector at runtime
 * */

template <typename T>
class Add
{
	T x;
	
	public:
		Add(T xx) : x(xx) { } 
		void operator()(T& e) const { e += x; }
};

struct PrintElm
{	
	void operator()(int & elm) const { std::cout << elm << ' ' ; }
};


// for another sample using bind2nd() with transform()
// outputs an array and outputs them in Print() function:
template <typename ForwardIter>
void Print2(ForwardIter first, ForwardIter last, const char* status)
{
	std::cout << status << std::endl;
	while ( first != last)
		std::cout << *first++ << " ";
	std::cout << std::endl;
	
};

int main() {
	float f = -123.45;
	absValue aObj;
	float abs_f = aObj(f);
	std::cout << "f =  " << f << " abs_f = " << abs_f << std::endl;
	std::cout << " aObj(45.f) : " << aObj(45.f) << std::endl;

	Line fa; 				// y = 1*x + 1
	Line fb(5.0, 10.0) ;  	// y = 5*x + 10 
	
	double y1 = fa(20.0); 	// y1 = 20 + 1
	double y2 = fb(3.0);	// y2 = 5*3 + 10
	
	std::cout << "y1 = " << y1 << " y2 = " << y2 << std::endl;


	std::vector<int> vect;
	for (int i = 1; i < 10; ++i) {
		vect.push_back(i);
	}	


	Print print_it;

	// the for_each function applied a specific function to each member of a range

	std::for_each( vect.begin(), vect.end(), print_it );
	std::cout << std::endl;

	// Function object using nontype template parameter
	int size = 5;
	std::vector<int> v(size);
	PrintElements<int> print_it2;
	std::for_each(v.begin(), v.end(), print_it2);
	std::for_each(v.begin(), v.end(), setValue<10>);
	std::for_each(v.begin(), v.end(), print_it2);
	
		
	std::vector<int> v2;
	for (int i = 0; i < size; i++) v2.push_back(i);

	std::for_each(v2.begin(), v2.end(), print_it2); std::cout << std::endl;
	std::for_each(v2.begin(), v2.end(), Add<int>(10));
	std::for_each(v2.begin(), v2.end(), print_it2); std::cout << std::endl;
	std::for_each(v2.begin(), v2.end(), Add<int>(*v2.begin()) );
	std::for_each(v2.begin(), v2.end(), print_it2);
	
	
	//
	// Predefined Functions Objects
	
	/*
	 * transform() algorithm combined elements from 2 vector containers by using 
	 * multiplies<int>() operation.  
	 * Then it write the results into the 3rd vector container
	 * */
	
	/* 
	 * Here are the 2 types of transform() algorithms:
	 * 1. Transforming elements
	 * OutputIterator
	 * transform ( InputIterator source.begin(), InputIterator source.end(), 
	 * 				OutputIterator destination.begin(), 
	 * 				UnaryFunc op )
	 * 
	 * 2. Combining elements of 2 sequences 
	 * OutputIterator
	 * transform ( InputIterator1 source1.begin(), InputIterator1 source1.end(),
	 * 				InputIterator2 source2.begin()
	 * 				OutputIterator destination.begin(),
	 * 				BinaryFunc op )	 
	 */
	 
	std::cout << std::endl;
	std::vector<int> v3;
	for (int i = 0; i < size ; i++) v3.push_back(i) ; 
	std::for_each(v3.begin(), v3.end(), print_it2); std::cout << std::endl;
	
	
	std::transform(v3.begin(), v3.end(), // source
					v3.begin(), 			// destination
					std::negate<int>() );
					
	std::for_each(v3.begin(), v3.end(), print_it2);  std::cout << std::endl;
	
	
	std::transform(v3.begin(), v3.end(),  	// source
					v3.begin(),					// second source
					v3.begin(), 				// destination
					std::multiplies<int>()); 	// operation
	
	std::for_each(v3.begin(), v3.end(), print_it2); std::cout << std::endl;
	
	
	//
	// bind2nd()
	/*
	 * What bogotobogo Hong says about bind2nd: 
	 * It returns function object with second parameter bound.  This function constructs (then)
	 * an UNARY function object from the BINARY function object op by 
	 * binding its second parameter to the fixed value x
	 * 
	 * The function object returned by bind2nd() has its operator() 
	 * defined such that it takes only 1 argument.  
	 * This argument is used to call binary function object op with x 
	 * as the fixed value for the second argument.  
	 * 
	 * */
	std::cout << "\n bind2nd() - examples " << std::endl;
	
	
	int size2 = 10;
	std::vector<int> v4;
	for (int i =0; i < size2; i++) v4.push_back(i) ;
	std::for_each( v4.begin(), v4.end(), PrintElm()); std::cout << std::endl;
	std::replace_if( v4.begin(), v4.end(), 
					std::bind2nd(std::equal_to<int>(), 0),
					101);
					
	std::for_each( v4.begin(), v4.end(), PrintElm() ); std::cout << std::endl;
	
	v4.erase( std::remove_if(v4.begin(), v4.end(), bind2nd( std::less<int>(), 3) ), 
			v4.end()
			);
	
	std::for_each(v4.begin(), v4.end(), PrintElm()); std::cout << std::endl;
	std::transform( v4.begin(), v4.end(), 
		std::ostream_iterator<int>(std::cout, " " ),
		std::negate<int>());
		
		
	// additional examples of bind2nd();
	
	// 1. To count all the elements within a vector that are less than or equal to 100,
	// use count_if()
	auto result = 
		std::count_if(v4.begin(), v4.end(), 
			std::bind2nd(std::less_equal<int>(), 100));

	std::cout << " result : " << result << std::endl;

	// 2. We can negate the binding of less_equal:
	// what it does is that each element will be tested to see if it is <=100
	// Then, the truth value of the result will be negated.  
	// Actually, the call counts those elements that are not <= 100
	
	auto result2 = 
		std::count_if(v4.begin(), v4.end(),
			std::not1(std::bind2nd(std::less_equal<int>(), 100) ));
			
			
	// another sample using bind2nd() with transform()
	// The code multiplies 10 to each element of an array and outputs them in Print() function:		
	int arr[] = {1, 2, 3, 4, 5};
			
	Print2(arr,arr+5, "Initial values");
	
	std::transform( arr, arr + 5, arr, std::bind2nd(std::multiplies<int>(), 10) );
	
	Print2(arr,arr+5, "New values: ");		
			

	return 0; 

}
