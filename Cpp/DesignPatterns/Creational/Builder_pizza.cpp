/**
 * 	@file 	Builder_pizza.cpp
 * 	@brief 	Creational Pattern, Builder, Pizza example 
 * 	@ref	https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns
 * 	@details Problem - Want: construct complex object, 
 * 		however, don't want to have complex constructor member or 1 that would need many arguments
 * 		Solution - Define intermediate object whose member functions define desired object 
 * 		part by part before object is available to client.  
 * 		Builder Pattern lets us defer construction of object until all options for creation specified
 * 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g stack_arr.c -o stack_arr
 * */
#include <string>
#include <iostream>
#include <memory>

// "Product"
class Pizza
{
	public:
		void setDough(const std::string& dough)
		{
			m_dough = dough;
		}
		void setSauce(const std::string& sauce)
		{
			m_sauce = sauce;
		}
		void setTopping(const std::string& topping)
		{
			m_topping = topping;
		}
		void open() const
		{
			std::cout << "Pizza with " << m_dough << " dough, " << m_sauce << " sauce and "
				<< m_topping << " topping.  Mmm." << std::endl; 
		}
	private:
		std::string m_dough;
		std::string m_sauce;
		std::string m_topping;
};

// "Abstract Builder"
class PizzaBuilder
{
	public:
		virtual ~PizzaBuilder() {};
		
		Pizza* getPizza()
		{
			m_pizza.release();
		}
		void createNewPizzaProduct()
		{
			m_pizza = std::make_unique<Pizza>();
		}
		virtual void buildDough() = 0;
		virtual void buildSauce() = 0;
		virtual void buildTopping() = 0;
	protected:
		std::unique_ptr<Pizza> m_pizza;
};

// --------------------------------------------------------------------

class HawaiianPizzaBuilder : public PizzaBuilder
{
	public:
		virtual ~HawaiianPizzaBuilder() {};
		
		virtual void buildDough()
		{
			m_pizza->setDough("cross");
		}
		virtual void buildSauce()
		{
			m_pizza->setSauce("mild");
		}
		virtual void buildTopping()
		{
			m_pizza->setTopping("ham+pineapple");
		}
};

class SpicyPizzaBuilder : public PizzaBuilder 
{
	public:
		virtual ~SpicyPizzaBuilder() {};
	
		virtual void buildDough()
		{
			m_pizza->setDough("pan baked");
		}
		virtual void buildSauce()
		{
			m_pizza->setSauce("hot");
		}
		virtual void buildTopping()
		{
			m_pizza->setTopping("pepperoni+salami");
		}
};

// --------------------------------------------------------------------

class Cook
{
	public:
		void openPizza()
		{
			m_pizzaBuilder->getPizza()->open();
		}
		void makePizza(PizzaBuilder* pb)
		{
			m_pizzaBuilder = pb;
			m_pizzaBuilder->createNewPizzaProduct();
			m_pizzaBuilder->buildDough();
			m_pizzaBuilder->buildSauce();
			m_pizzaBuilder->buildTopping();
		}
	private:
		PizzaBuilder* m_pizzaBuilder;
};

int main()
{
	Cook cook;
	HawaiianPizzaBuilder 	hawaiianPizzaBuilder;
	SpicyPizzaBuilder 		spicyPizzaBuilder;
	
	cook.makePizza(&hawaiianPizzaBuilder);
	cook.openPizza();
	
	cook.makePizza(&spicyPizzaBuilder);
	cook.openPizza();
}

 
	
	
	
