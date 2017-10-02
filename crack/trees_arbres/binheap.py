"""
	@file   : binheap.py
	@brief  : Binary Heap 
    @details : BinHeap
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170918
	@ref    : cf. http://interactivepython.org/runestone/static/pythonds/Trees/SearchTreeImplementation.html 
                     https://github.com/bnmnetp/pythonds/blob/master/trees/bst.py
  
	If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
	
	https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
	
	which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
	Otherwise, I receive emails and messages on how all my (free) material on 
	physics, math, and engineering have helped students with their studies, 
	and I know what it's like to not have money as a student, but love physics 
	(or math, sciences, etc.), so I am committed to keeping all my material 
	open-source and free, whether or not 
	sufficiently crowdfunded, under the open-source MIT license: 
	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
	Just don't be an asshole and not give credit where credit is due.  
	Peace out, never give up! -EY
"""  
class BinHeap: 
    def __init__(self):
        self.heapList = [0] # 1-based counting
        self.currentSize = 0 

    def buildHeap(self,alist):
        """ @fn  buildHeap(self,alist)
            @brief  builds a new heap from a list of keys
        """
        i = len(alist) // 2 # integer division is //
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        print(len(self.heapList),i)
        while (i>0): 
            print(self.heapList,i)
            self.percDown(i)
            i=i-1
        print(self.heapList,i)

    def percDown(self,i):
        while(i*2) <= self.currentSize: # check with i*2 because we need to know if i has children or not 
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]: # violates heap order property that parent is less than child in value         
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc] # do the switch between them
                self.heapList[mc] = tmp 
            i = mc # check now the children of the child of i 
    
    def minChild(self,i):
        """ @fn  minChild
            @brief  finding the minimum amongst the children of parent ith node
        """
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i*2] < self.heapList[i * 2  + 1]:
                return i * 2
            else:
                return i * 2 + 1 

    def percUp(self,i):
        while i // 2 > 0:   # i // 2 is integer division (node of parent of ith node)
            if self.heapList[i] < self.heapList[i//2]: # violates heap order property
                tmp = self.heapList[i//2] # parent is now tmp
                self.heapList[i//2] = self.heapList[i] # percolate up the ith node to where its parent was
                self.heapList[i] = tmp # put parent in where i was 
            i = i // 2 # keep doing this check for heap order property violations  

    def insert(self,k):
        """ @fn  insert
            @brief adds new item to the heap
        """
        self.heapList.append(k) 
        self.currentSize = self.currentSize + 1 
        self.percUp(self.currentSize) # check for heap order property violations

    def delMin(self):
        """ @fn  delMin
            @brief  returns item with minimum key value, leaving item in the heap

            @note Since heap order property requires root of tree be smallest item in tree, 
                    finding the minimum item is easy.  Hard part is restoring full compliance 
                    with heap structure and heap order property.  
        """
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1 # take last item in list and move it to root position
        self.heapList.pop()
        self.percDown(1)
        return retval
    
    def isEmpty(self):
        """ @fn  isEmpty
            @brief  returns true if the heap is empty, false otherwise
        """
        if currentSize == 0:
            return True
        else:
            return False  

