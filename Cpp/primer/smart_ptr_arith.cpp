/**
 * @file   : smart_ptr_arith.cpp
 * @brief  : Smart pointer (shared and unique ptrs) arithmetic, in C++14  
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170904
 * @ref    : http://en.cppreference.com/w/cpp/memory/unique_ptr
 *         : Stanley B. Lippman, Josee Lajoie, Barbara E. Moo.  C++ Primer, Fifth Edition.  Sec. 3.5, Ch. 12 
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
 * g++ -std=c++14 smart_ptr_arith.cpp -o smart_ptr_arith.exe
 * -std=c++14 flag needed for auto, etc.
 * */
#include <iostream> // std::cout 
#include <memory>  // std::shared_ptr, std::unique_ptr std::make_unique (C++14)


template<typename T>
struct array_deleter
{
    void operator()(T const* p)
    {
        delete [] p;
    }
};

int main(int argc, char* argv[]) {
    constexpr const size_t Lx = {1<<5};  // 2^5 = 32
    std::cout << " Lx : " << Lx << std::endl;

    // shared_ptr
    //  Initialization
    std::shared_ptr<float> sp( new float[Lx]  );    
    std::shared_ptr<float> spb( new float[Lx],array_deleter<float>()  );    
    // with C++11, you can use std::default_delete partial specialization for array types instead of array_deleter
    std::shared_ptr<float> spc(new float[Lx],std::default_delete<float[]>());
    // use lambda expression instead of functors
    std::shared_ptr<float> spd(new float[Lx],[](float* p) { delete[] p; });

    // "boilerplate" initialization of interesting values
    for (auto iptr = spc.get(); iptr != spc.get() + Lx/2; iptr++) {
        *iptr = 11.f * ((int) (iptr - spc.get()));
    }

    for (float* b = spc.get(); b < spc.get() + Lx; b++) {
        std::cout << " b : " << b << " *b : " << *b << std::endl;
    }

    for (auto iptr = spc.get()+Lx/2; iptr != spc.get() + Lx; iptr++) {
        *iptr = 0.11f * ((int) (iptr - spc.get()));
    }

    for (float* b = spc.get(); b < spc.get() + Lx; b++) {
        std::cout << " b : " << b << " *b : " << *b << std::endl;
    }
    // array index access
    for (int idx = 0 ; idx < Lx; idx++) {
        std::cout << " " << idx << " : " << spc.get()[idx] << ", ";
    }


    // unique_ptr, initialization

    auto uptr11 = std::unique_ptr<float[]>{ new float[Lx]};  // this will correctly call delete []
//    auto uptr14 = std::make_unique<float[]>(Lx);

    // with custom deleter/deconstructor
    auto deleter = [&](float* ptr){ delete[] ptr; }; 
    std::unique_ptr<float[], decltype(deleter)> up(new float[Lx], deleter);

    // "boilerplate" initialization of interesting values
    for (auto iptr = up.get(); iptr != up.get() + Lx/2; iptr++) {
        *iptr = 101.f * ((int) (iptr - up.get()));
    }

    for (float* b = up.get(); b < up.get() + Lx; b++) {
        std::cout << " b : " << b << " *b : " << *b << std::endl;
    }

    for (auto iptr = up.get()+Lx/2; iptr != up.get() + Lx; iptr++) {
        *iptr = 0.011f * ((int) (iptr - up.get()));
    }

    for (float* b = up.get(); b < up.get() + Lx; b++) {
        std::cout << " b : " << b << " *b : " << *b << std::endl;
    }
    // array index access
    for (int idx = 0 ; idx < Lx; idx++) {
        std::cout << " " << idx << " : " << up.get()[idx] << ", ";
    }



    return 0;
}
