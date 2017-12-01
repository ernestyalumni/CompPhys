# Design Patterns  

cf. [virtual function = 0, `cplusplus.com`](http://www.cplusplus.com/forum/beginner/40226/)

**pure virtual** or **abstract function** is a function, required to be overwritten in a derived class.  

e.g. 

``` 
class Body {
	public:
		virtual double Volume() = 0;
		virtual double Surface() = 0;
}
```  

