/**
 * @file   : commonmultiplestreams.cu
 * @brief  : Common pattern for dispatching CUDA operations to multiple streams 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170904  
 * @ref    : http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
 * 		   : 3.2.5. Asynchronous Concurrent Execution of CUDA Toolkit v8.0, 3. Programming Interface  
 * 		   : John Cheng, Max Grossman, Ty McKercher. Professional CUDA C Programming. 1st Ed. Wrox. 2014
 * 		   : Ch. 6 Streams and Concurrency; pp. 271   
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
/* 
 * COMPILATION TIP
 * nvcc -std=c++11 smart_ptrs_arith.cu -o smart_ptrs_arith.exe
 * 
 * */
#include <iostream>

int main(int argc, char *argv[]) {


}
