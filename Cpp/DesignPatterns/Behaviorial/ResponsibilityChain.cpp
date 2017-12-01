/**
 * 	@file 	ResponsibilityChain.cpp
 * 	@brief 	Behaviorial Pattern, Chain of Responsibility 
 * 	@ref	https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns
 * 	@details Chain of Responsibility intends to avoid coupling sender of a request to its receiver by 
 * 	giving more 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g ResponsibilityChain.cpp -o ResponsibilityChain
 * */
#include <iostream>

class Handler {
	protected:
		Handler *next;
		
	public:
		Handler() {
			next = NULL;
		}
		
		virtual ~Handler() { }
		
		virtual void request(int value) = 0;
		
		void setNextHandler(Handler *nextInLine) {
			next = nextInLine;
		}
};

class SpecialHandler : public Handler {
	private:
		int myLimit;
		int myId;
		
	public:
		SpecialHandler(int limit, int id) {
			myLimit = limit;
			myId = id;
		}
		
		~SpecialHandler() { }
		
		void request(int value) {
			if(value < myLimit) {
				std::cout << "Handler " << myId << " handled the request with a limit of " << myLimit << std::endl; 
			} else if (next != NULL) {
				next->request(value);
			} else {
				std::cout << "Sorry, I am the last handler (" << myId << ") and I can't handle the request." << std::endl; 
			}
		}
};

int main() {
	Handler *h1 = new SpecialHandler(10, 1);
	Handler *h2 = new SpecialHandler(20, 2); 
	Handler *h3 = new SpecialHandler(30, 3); 

	h1->setNextHandler(h2);
	h2->setNextHandler(h3);
	
	h1->request(18); 
	
	h1->request(40);
	
	delete h1;
	delete h2;
	delete h3;
	
	return 0;
}
				
