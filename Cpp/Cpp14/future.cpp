/**
 * 	@file 	future.cpp
 * 	@brief 	C++ program to demonstrate future   
 * 	@ref	http://en.cppreference.com/w/cpp/thread/future 
 * http://www.cplusplus.com/reference/future/future/
 * 	@details object that can retrieve value from some provider object or function, properly synchronizing this access if in different threads  
 * 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g future.cpp -o future
 * */
#include <iostream>

#include <future> 

#include <thread>  

int main() 
{
	// future from a packaged_task  
	std::packaged_task<int()> task([]{ return 7; }); 	// wrap the function  
	std::future<int> f1 = task.get_future(); 			// get a future
//	std::thread t(std::move(task));						// launch on a thread // undefined reference to `pthread_create'
	
	// future from an async() 
//	std::future<int> f2 = std::async(std::launch::async, []{ return 8; }); // undefined reference to `pthread_create'
	
	// future from a promise
	std::promise<int> p;
	std::future<int> f3 = p.get_future();
//	std::thread( [&p]{ p.set_value_at_thread_exit(9); }).detach();  // undefined reference to `pthread_create'
	
	std::cout << "Waiting..." << std::flush;
	f1.wait();
//	f2.wait();
	f3.wait();
	std::cout << "Done!\nResults are: " 
				<< f1.get() << ' ' << //f2.get() << ' ' << f3.get() << '\n';  
					f3.get() << '\n';
//	t.join(); 


}	
