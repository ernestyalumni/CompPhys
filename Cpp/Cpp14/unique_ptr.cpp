/**
 * @file   : unique_ptr.cpp
 * @brief  : demonstrate examples, usage of unique_ptr, in C++14  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170901
 * @ref    : http://en.cppreference.com/w/cpp/memory/unique_ptr
 *         : Stanley B. Lippman, Josee Lajoie, Barbara E. Moo.  C++ Primer, Fifth Edition.  
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* on g++ 5.3.1.
 * g++ -std=c++14 unique_ptr.cpp -o unique_ptr.exe
 * -std=c++14 flag needed for auto, etc.
 * */

#include <iostream> // std::cout 
#include <memory>  // std::unique_ptr std::make_unique (C++14)

#include <vector> // std::vector
#include <cassert> // assert

#include <fstream> // std::ofstream

#include <string> // std::string

// cf. http://en.cppreference.com/w/cpp/utility/forward
#include <utility> // std::forward

struct B {
    // virtual specifier specifies that non-static member function is virtual and supports dynamic binding
    virtual void bar() { std::cout << "B::bar\n"; }
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

// 12.1.5. unique_ptr of Lippman, Lajoie, Moo
std::unique_ptr<int> clone(int p) {
    // ok: explicitly create a unique_ptr<int> from int*
    return std::unique_ptr<int> (new int(p));
};

int main()
{
    std::cout << "unique ownership semantics demo\n";  
    { 
        auto p = std::make_unique<D>(); // p is a unique_ptr that owns a D
        auto q = pass_through(std::move(p));
        assert(!p);     // now p owns nothing and holds a null pointer
        q->bar();       // and q owns the D object
    }   // ~D called here


    std::cout << "Runtime polynorphism demo\n";
    {
        std::unique_ptr<B> p = std::make_unique<D>();   // p is a unique ptr that owns a D
                                                        // as a pointer to base
        p->bar();   // virtual dispath
        
        std::vector<std::unique_ptr<B>> v;              // unique_ptr can be stored in a container
        v.push_back(std::make_unique<D>());
        v.push_back(std::move(p));
        v.emplace_back(new D);
        for (auto& p: v) p->bar(); // virtual dispatch
    } // ~D called 3 times

    std::cout << "Custom deleter demo\n";
    std::ofstream("demo.txt") << 'x'; // prepare the file to read
    {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> fp(std::fopen("demo.txt", "r"), 
                                                                &std::fclose);
        if(fp) // fopen could have failed; in which case fp holds a null pointer
            std::cout << (char)std::fgetc(fp.get()) << '\n';
    }   // fclose() called here, but only if FILE* is not a null pointer
        // (that is, if fopen succeeded)
    
    std::cout << "Custom lambda expression deleter demo\n";
    {
        std::unique_ptr<D, std::function<void(D*)>> p(new D, [](D* ptr) 
            {
                std::cout << "destroying from a custom deleter ... \n";
                delete ptr;
            });     // p owns D
        p->bar();
    }   // the lambda above is called and D is destroyed  

    std::cout << "Array form of unique_ptr demo\n";
    {
        std::unique_ptr<D[]> p(new D[3]);
    }   // calls ~D 3 times

    /**
      * from Lippman, Lajoie, Moo, Ch. 12 12.1.5 unique_ptr 
      */ 
    std::unique_ptr<double> p1; // unique_ptr that can point at a double
    std::unique_ptr<int> p2( new int(42)); // p2 points to int with value 42
    std::unique_ptr<float> p2b(new float[42]);
    std::cout << " *p2b : " << *p2b << std::endl;

    /* instead, do this for an array:  
     * cf. https://stackoverflow.com/questions/21377360/proper-way-to-create-unique-ptr-that-holds-an-allocated-array
     */
    auto p2c = std::unique_ptr<float[]>{ new float[42]};
    // std::cout << " p2c : " << p2c << std::endl; // note: cannot convert `p2c` type to ... template argument deduction failed
    std::cout << " p2c.get() : " << p2c.get() << std::endl; 
    std::cout << " p2c[0] : " << p2c[0] << std::endl; 
    for (int i =0; i<42 ; i++) {
        std::cout << p2c[i] ;
        p2c[i] = i*10;    
        std::cout << " " << i << " : " << p2c[i] << " " ;
    }   

    /* 
     * Because unique_ptr owns object to which it points, unique_ptr doesn't support ordinary copy or assignment  
     */ 
    std::unique_ptr<std::string> p1b(new std::string("Stegosaurus"));
//    std::unique_ptr<string> p2c(p1b); // error
//    std::unique_ptr<std::string> p3;
//    p3 = p2c;
//    p3 = p1b; // error use of deleted function  

    std::cout << "\n *p1b.get() : " << *p1b.get() << std::endl;

    /*
     * We can't copy or assign unique_ptr, we can transfer ownership from 1 (nonconst) unique_ptr to another 
     * by calling release or reset */
    // transfers ownership from p1 (which points to the string Stegosaurus) to p2
    std::unique_ptr<std::string> p2d(p1b.release());  // release makes p1b null

//    std::cout << "\n p1b : " << *p1b.get() << std::endl; // Segmentation fault
    std::cout << "\n p1b.get() : " << p1b.get() << std::endl;
    std::cout << "\n p2d : " << *p2d.get() << std::endl;
    

    std::unique_ptr<std::string> p3(new std::string("Trex"));
    // transfers ownership from p3 to p2d
    p2d.reset(p3.release()); // reset deletes the memory to which p2d had pointed

    auto p3b = clone(5);
    std::cout << " p3b : " << *p3b.get() << std::endl;     

    // Passing a deleter to unique_ptr

    auto uptr11 = std::unique_ptr<float[]>{ new float[42]};
    auto uptr14 = std::make_unique<float[]>(42);
    for (int i =0; i<42 ; i++) {
        std::cout << uptr11[i] << " " << uptr14[i] << " ";
        uptr11[i] = i*11.f;
        uptr14[i] = i*140.f;    
        std::cout << " " << i << " : " << uptr11[i] << " " << uptr14[i]  << std::endl;
    }   

    uptr11.get() + 5;
    std::cout << " uptr11.get() +5 : " << uptr11.get() +5 << std::endl;
    std::cout << " *uptr11.get() : " << *uptr11.get() << std::endl;

    std::shared_ptr<float> shptr11{ std::move(uptr11.get() + 5 ) }; 


}    