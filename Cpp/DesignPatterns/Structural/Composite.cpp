/**
 * 	@file 	Composite.cpp
 * 	@brief 	Structural Pattern, Composite example 
 * 	@ref	https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns
 * 	@details Composite lets clients treat individual objects and compositions of objects uniformly.   
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g Composite.cpp -o Composite
 * */
#include <vector>
#include <iostream> // std::cout 
#include <memory> // std::auto_ptr
#include <algorithm> // std::for_each

class Graphic
{
	public:
		virtual void print() const = 0;
		virtual ~Graphic() {}
}; 

class Ellipse : public Graphic {
	public:
		void print() const {
			std::cout << "Ellipse " << std::endl; 
		}
};

class CompositeGraphic : public Graphic {
	public:
		void print() const {
			for(Graphic * a: graphicList_) {
				a->print();
			}
		}
		
		void add(Graphic *aGraphic) {
			graphicList_.push_back(aGraphic);
		}
		
	private:
		std::vector<Graphic*> graphicList_;
};

int main() {
	// Initialize 4 ellipses 
	const std::auto_ptr<Ellipse> ellipse1(new Ellipse());
	const std::auto_ptr<Ellipse> ellipse2(new Ellipse());
	const std::auto_ptr<Ellipse> ellipse3(new Ellipse());
	const std::auto_ptr<Ellipse> ellipse4(new Ellipse());
	
	// Initialize 3 composite graphics
	const std::auto_ptr<CompositeGraphic> graphic(new CompositeGraphic());
	const std::auto_ptr<CompositeGraphic> graphic1(new CompositeGraphic());
	const std::auto_ptr<CompositeGraphic> graphic2(new CompositeGraphic());
	
	// Composes the graphics
	graphic1->add(ellipse1.get());
	graphic1->add(ellipse2.get());
	graphic1->add(ellipse3.get());
	
	graphic2->add(ellipse4.get());
	
	graphic->add(graphic1.get());
	graphic->add(graphic2.get());
	
	// Prints the complete graphic (4 times the string "Ellipse")
	graphic->print();
	return 0;
}
