/**
 * 	@file 	Factory_pizza.cpp
 * 	@brief 	Creational Pattern, Builder, Pizza example 
 * 	@ref	https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns
 * 	@details Problem - Want: decide at run-time what object to be created based on some configuration or application parameter.  
 * 		When we write the code, we don't know what class should be instantiated.  
 * 		Solution - Define interface for creating an object, but let subclasses decide which class to instantiate.  
 * 			Factory Method lets class defer instantiation to subclasses.  
 *  
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g Factory_pizza.cpp -o Factory_pizza
 * */
#include <stdexcept>
#include <iostream>
#include <memory>
 
class Pizza {
	public:
		virtual int getPrice() const = 0;
		virtual ~Pizza() {}; /* without this, no destructor for derived Pizza's will be called. */
};

class HamAndMushroomPizza : public Pizza {
	public:
		virtual int getPrice() const { return 850; };
		virtual ~HamAndMushroomPizza() {};
};

class DeluxePizza : public Pizza {
	public:
		virtual int getPrice() const { return 1050; };
		virtual ~DeluxePizza() {};
};

class HawaiianPizza : public Pizza {
	public:
		virtual int getPrice() const { return 1150; };
		virtual ~HawaiianPizza() {};
};

class PizzaFactory {
	public:
		enum PizzaType {
			HamMushroom, 
			Deluxe,
			Hawaiian
		};
			
	static std::unique_ptr<Pizza> createPizza(PizzaType pizzaType) {
		switch (pizzaType) {
			case HamMushroom: 	return std::make_unique<HamAndMushroomPizza>();
			case Deluxe:		return std::make_unique<DeluxePizza>();
			case Hawaiian:		return std::make_unique<HawaiianPizza>();
		}
		throw "invalid pizza type.";
	}
};

/*
 * Create all vailable pizzas and print their prices
 * */

void pizza_information(PizzaFactory::PizzaType pizzatype)
{
	std::unique_ptr<Pizza> pizza = PizzaFactory::createPizza(pizzatype);
	std::cout << "Price of " << pizzatype << " is " << pizza->getPrice() << std::endl; 
}

int main()
{
	pizza_information(PizzaFactory::HamMushroom);
	pizza_information(PizzaFactory::Deluxe);
	pizza_information(PizzaFactory::Hawaiian); 
}
