/**
 * 	@file 	Factor_Computer.cpp
 * 	@brief 	Creational Pattern, Factory, Computer example 
 * 	@ref	https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns
 * 	@details Problem - Want: decide at run time what object to be created based on some configuration or application parameter.  
 * 	When we write code, we don't know what class shouldbe instantiated.  
 * 	Solution - Define interface for creating an object, but let subclasses decide which class to instantiate.  
 * 		Factory Method lets class defer instantiation to subclasses.  
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g Factory_Computer.c -o Factory_Computer
 * */
 #include <string>
 
 
class Computer
{
	public:
		virtual void Run() = 0;
		virtual void Stop() = 0;
		
		virtual ~Computer() {}; /* without this, you do not call laptop or Desktop destructor in this example! */ 
};

class Laptop: public Computer
{
	public:
		void Run() override {mHibernating = false; };
		void Stop() override {mHibernating = true; };
		virtual ~Laptop() {}; /* because we have virtual functions, we need virtual destructor */ 
	private:
		bool mHibernating; // Whether or not the machine is hibernating
};

class Desktop: public Computer
{
	public:
			void Run() override {mOn = true; };
			void Stop() override {mOn= false; };
			virtual ~Desktop() {}; 
	private:
		bool mOn;	// Whether or not the machine has been turned on
}; 

class ComputerFactory
{
	public:
		static Computer *NewComputer(const std::string &description)
		{
			if (description == "laptop")
				return new Laptop;
			if (description == "desktop") 
				return new Desktop;
			return NULL;
		}
}; 


int main() {
	auto test_new_computer = ComputerFactory();
	auto test_new_laptop = test_new_computer.NewComputer("laptop");

	test_new_laptop->Run();
	test_new_laptop->Stop();
}
