/** 
 * @file   : unique_ptr.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Simple examples for shared pointers, for C++11, C++14; 
 * 			try playing around with the compilation standard flag, -std=c++XX 
 * 			where XX=11 or 14, and see how it works (or doesn't) 
 * @ref    : http://en.cppreference.com/w/cpp/memory/unique_ptr
           : Stanley B. Lippman, Josée Lajoie, Barbara E. Moo.  
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
 **/

#include <iostream> // std::cout  
#include <memory>   // std::unique_ptr, std::make_unique (since C++14)

#include <cassert> // assert

struct B {
    virtual void bar() { 
        std::cout << "B::bar \n"; 
    }
    virtual ~B() = default;    
};
struct D : B 
{
    D() { std::cout << "D::D\n"; }
    ~D() { std::cout << "D::~D\n"; } 
    void bar() override { std::cout << "D::bar\n"; }
}; 

// a function consuming a unique_ptr can take it by value or by rvalue reference 
std::unique_ptr<D> pass_through(std::unique_ptr<D> p)
{
    p->bar();
    return p;
}

int main(int argc, char* argv[]) {
    std::cout << "unique ownership semantics demo\n";
    { 
        auto p = std::make_unique<D>(); // p is a unique_ptr that owns a D 
        auto q = pass_through(std::move(p)); 
        assert(!p); // now p owns nothing and holds a nully pointer 
        q->bar();   // and q owns the D object
    } // ~D called here  

    std::cout << "Stepping through once more ... \n";
    { 
        auto p = std::make_unique<D>(); // p is a unique_ptr that owns a D 
        auto q = pass_through(std::move(p)); 
        assert(!p); // now p owns nothing and holds a nully pointer 
        
        q->bar();   // and q owns the D object
    } // ~D called here  



}