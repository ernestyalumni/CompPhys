/** 
 * @file   : sh_ptr.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Simple examples for shared pointers, for C++11, C++14; 
 * 			try playing around with the compilation standard flag, -std=c++XX 
 * 			where XX=11 or 14, and see how it works (or doesn't) 
 * @ref    : Stanley B. Lippman, Josée Lajoie, Barbara E. Moo.  
           : C++ Primer (5th Edition) 5th Edition.  Addison-Wesley Professional; 
           : 5 edition (August 16, 2012).  ISBN-13: 978-0321714114.  
 *
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
 /**
 * Compilation tips
 *
 * g++ -std=c++14 ptrs_unique_shared.cpp -o ptrs_unique_shared.exe
 * g++ -std=c++11 ptrs_unique_shared.cpp -o ptrs_unique_shared.exe
 **/

#include <iostream> // std::cout  
#include <memory>   // std::shared_ptr, std::make_shared, std::unique_ptr

#include <string>   // std::string
#include <list>     // std::list
#include <vector>   // std::vector

class StrBlob {
    public:
        using size_type = std::vector<std::string>::size_type; 
        
        // constructors
        StrBlob();
        StrBlob(std::initializer_list<std::string> il); 
        size_type size() const {
            return data->size();
        }

        bool empty() const {
            return data->empty(); 
        }

        // add and remove elements
        void push_back(const std::string &t) {
            data->push_back(t);            
        }

        void pop_back();
        // element access 

        std::string& front();
        std::string& back();  

    private:
        std::shared_ptr<std::vector<std::string>> data;
        // throws msg if data[i] isn't valid 
        void check(size_type i, const std::string &msg) const; 
};

/**
  * ** In order to correctly use shared_ptr with an array, 
  * ** you must supply a custom deleter 
  */
template<typename T>
struct array_deleter
{
    void operator()(T const* p)
    {
        delete [] p;
    }
};


int main(int argc, char* argv[]) {
    std::shared_ptr<std::string> p1;  // shared ptr that can point at a string
    std::shared_ptr<std::list<int>> p2; // shared_ptr that can point a list of ints 

    // shared_ptr that points to an int with value 42 
    std::shared_ptr<int> p3     = std::make_shared<int>(42);
    // p4 points to a string with value 9999999999
    std::shared_ptr<std::string> p4  = std::make_shared<std::string>(10, '9'); 

    std::cout << " p1 itself (true if p1 points to an object : " << p1 << std::endl; 
    std::cout << " p2 itself (true if p2 points to an object : " << p2 << std::endl; 

    std::cout << " p3 itself (true if p3 points to an object : " << p3 << std::endl; 
    std::cout << " p4 itself (true if p4 points to an object : " << p4 << std::endl; 
 
    // *p deference p to get object to which p points 
    std::cout << " *p3  : " << *p3 << std::endl; 
    std::cout << " *p4  : " << *p4 << std::endl; 
    
    // returns true if p.use_count() is 1; false otherwise  
    std::cout << " p3.unique()  : " << p3.unique() << std::endl; 
    std::cout << " p4.unique()  : " << p4.unique() << std::endl; 
    
    // p5 points to an int that is value initialized (Sec. 3.3.1 (p. 98)) to 0
    std::shared_ptr<int> p5 = std::make_shared<int>(); 

    // p6 points to a dynamically allocated, empty vector <string> 
    auto p6 = std::make_shared<std::vector<std::string>>();  

    /**
      * EY : 20170831 but is p6 allocated on the "stack" or the "heap"?  I want to have 
      * access to the larger free pool of memory allowed for by the heap, I believe 
      */ 
    
    /** 
      ** Copying and Assigning shared_ptrs  **  
      */  
    auto p = std::make_shared<int>(42); // object to which p points has 1 user
    std::cout << " Beforehand, p.use_count() : " << p.use_count() << std::endl; 
    
    auto q(p);  // p and q point to the same object 
                // object to which p and q point has 2 users  
    
    std::cout << " After pointing, q.use_count() : " << q.use_count() << std::endl;         
    std::cout << " After pointing, p.use_count() : " << p.use_count() << std::endl;         
    
    auto r = std::make_shared<int>(42); // int to which r points has 1 user 
    r = q;  // assign to r, making it point to a different address 
            // increase the use count for the object to which q points 
            // reduce the use count of the object to which r had pointed
            // the object r had pointed to has no users; that object is 
            // automatically freed
    
    std::cout << " q.use_count() : " << q.use_count() << std::endl; 
    std::cout << " r.use_count() : " << r.use_count() << std::endl; 
    
    
    /**
      * vector "owns" its own elements  
      */
    
    std::vector<std::string> v1; // empty vector 
    {   // new scope
        std::vector<std::string> v2 = {"a", "an", "the"};
        v1 = v2; // copies the elements from v2 into v1 
    }   // v2 is destroyed, which destroys the elements in v2 
        // v1 has 3 elements, which are copies of the ones originally in v2 

    /* In general, when 2 objects share the same underlying data, 
     * we can't unilaterally destroy the data when an object of that type goes away:
     */ 
    
 //   auto alStrBlob = StrBlob();

    /**
      * cf. https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used
      */  
    
    std::shared_ptr<int> sp( new int[10], array_deleter<int>() );

    std::cout << " *sp : " << *sp << std::endl;  // 0  
 
    // p.get() returns the pointer in p 
    sp.get();

    std::cout << " sp.get() : " << sp.get() << std::endl;  // 0  

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp.get()[" << idx << "] : " << sp.get()[idx] << std::endl; 
        sp.get()[idx] = idx + 1;
    }

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp.get()[" << idx << "] : " << sp.get()[idx] << std::endl; 
        sp.get()[idx] = idx + 1;
    }

    /**
     * with C++11, you can use 
     * std::default_delete 
     * partial specialization for array types instead of array_deleter above. 
     */
    std::shared_ptr<int> sp1( new int[10], std::default_delete<int[]>() );

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp1.get()[" << idx << "] : " << sp1.get()[idx] << std::endl; 
        sp1.get()[idx] = idx + 2;
    }

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp1.get()[" << idx << "] : " << sp1.get()[idx] << std::endl; 
        sp1.get()[idx] = idx + 2;
    }
    
    /** 
     * You can also use a lambda expression instead of functors.  
     */
    std::shared_ptr<int> sp2( new int[10], []( int *p) { delete[] p; } );

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp2.get()[" << idx << "] : " << sp2.get()[idx] << std::endl; 
        sp2.get()[idx] = idx + 3;
    }

    for (int idx=0; idx< 10; idx++) {
        std::cout << " sp2.get()[" << idx << "] : " << sp2.get()[idx] << std::endl; 
        sp2.get()[idx] = idx + 3;
    }

    /**
     * unless you actually need to share the managed object, 
     * unique_ptr better suited for this task, 
     * since it has a partial specialization for array types
     */ 
    std::unique_ptr<int[]> up( new int[10] ); // this will correctly call delete[]
    
    for (int idx=0; idx< 10; idx++) {
        std::cout << " up.get()[" << idx << "] : " << up.get()[idx] ; 
        up.get()[idx] = idx + 5;
    }

    for (int idx=0; idx< 10; idx++) {
        std::cout << " up.get()[" << idx << "] : " << up.get()[idx] ; 
        up.get()[idx] = idx + 5;
    }
    


    

}