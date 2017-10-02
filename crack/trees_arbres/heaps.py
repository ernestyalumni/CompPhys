"""
	@file   : heaps.py
	@brief  : (Binary) Heap 
    @details : minHeap maxHeap minheapsort maxheapsort 
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
class minHeap:
    """ @class minHeap
        @brief minimum value is at the root, always
        @details 1-based counting.
                remember that, for all node i=1,2,...L
                p = i // 2 # (parent)
                l = 2*i # left child of ith node
                r = 2*i + 1 # right child of ith node
                min heap condition:  
                A(p(i)) <= A(i)
    """
    def __init__(self):
        """
        @param heaplst - heap List, with 1 based counting
        """ 
        self.heaplst = [0]
        self.size = 0 # L

    def percup(self,i):
        """ @func  percp
            @brief traverse tree upwards and check min heap property; if violated, switch
            @param i=1,2,...L
        """
        p = i // 2
        while (p > 0): # check if ith node has a parent or not 
            if ( self.heaplst[p] > self.heaplst[i]): # violation of min heap property, do a switch
                tmp = self.heaplst[p]
                self.heaplst[p] = self.heaplst[i]
                self.heaplst[i] = tmp
            i = i // 2 # percolate upwards to the parent of ith node
            p = i // 2 
    
    def insert(self,val):
        """ @fn inser(self,val)
            @brief insert a value into the heap
            @details insert a value into the heap, at the end of the heap; 
                        then use percup to check min heap condition 
            @returns nothing
        """
        self.heaplst.append(val)
        self.size += 1 
        self.percup(self.size)  # check for heap order property violations

    def minChild(self,i):
        """ @fn minChild(self,i)
            @brief find the child (of parent at i) with smallest value, i.e. A(2*i) or A(2*i+1) smaller?
            @returns index of the smaller child of i 
        """
        l = 2*i
        r = 2*i + 1
        if (r > self.size): # there's only a left child; no right child
            return l  
        else:
            if (self.heaplst[l] < self.heaplst[r]):
                return l
            else:
                return r

    def percdown(self,i):
        """ @fn percdown(self,i)
            @brief enforces the min heap property, starting from node i, and switches node i if necessarily "downward"        
            @returns nothing
        """
        l = 2*i # left child of i 
        while (l <= self.size): # check to see if node i even has a child or not 
            mc = self.minChild(i) # obtain child with smallest value 
            if self.heaplst[i] > self.heaplst[mc]: # violation of min heap property, node i's value is bigger than 1 of its child, its smallest child 
                tmp = self.heaplst[mc] 
                self.heaplst[mc] = self.heaplst[i] # swap the node i down to its child
                self.heaplst[i] = tmp
            i = mc # percolate node i down to its smallest child
            l = 2*i # recalculate left child to see if i does have a left child in the next iteration 

    def delMin(self):
        """ @fn delMin(self)
            @brief deletes the minimum value of the binary heap, which would be A(1), and maintain min heap property
            @note easiest choice for a value to replace A[1] is the very last value at the end of the heap, A[L] because
                    otherwise you'd have to shift or swap 2 times at least
            @returns minimum value, A[1]
        """
        L = self.size
        minimum_val = self.heaplst[1]  
        self.heaplst[1] = self.heaplst[L] # swap last in heap list to be first, so to avoid further swapping 
        self.size = self.size-1 # the heap shrank in size by 1
        self.heaplst.pop() # remove the last element, since it's now the first
        self.percdown(1) # enforce min heap property
        return minimum_val

    def buildHeap(self,X):
        """ @fn  buildHeap
            @brief build heap, percolating down or heapify at every node starting from i=L//2
                    start here because every node L//2 +1, L//2 +2... is a leaf  
            @param X list to add 
        """
        self.heaplst = [0,] + X[:]
        self.size = len(X)
        i = len(X) // 2 # start from L/2 because L/2+1,L/2+2 ... are leafs 
        print(len(X), len(self.heaplst),i)
        while (i>0): 
            print(self.heaplst,i)
            self.percdown(i)
            i = i-1
        print(self.heaplst,i)

class maxHeap(object):
    """ @class maxHeap
        @brief binary heap that satisfies the max heap property 
        @note 1-based counting 
            Given some X:{1,2,...L}, we want A:{1,2,...L}, with values in X(i)'s s.t. 
            if p is for parent 
            p:{2,3,...L}->{1,2,...L/2}
            p(i) = i/2
            if l is for left child, l(i) = 2*i ; l:{1,2,...L/2}->{2,4,..2*(L/2)}
            if r is for right child, r(i) = 2*i +1; r:{1,2...L/2}->{3,5,...2*(L/2)+1 <= L }
            max heap property: A[p(i)] >= A[i] \forall \, i=2,3,...L
    """
    def __init__(self):
        self.heaplst = [0]  # heap List, put in dummy 0 for 1-based counting
        self.size = 0 # L

    def percup(self,i):
        """ @fn percup
            @brief percolate up; check for max heap property from ith node; 
            @note max heap property only checks between parent and child, not amongst children
        """
        p = i // 2 # parent 
        while (p > 0): # make sure a parent exists 
            if (self.heaplst[p] < self.heaplst[i]): # if max heap property is violated, 
                tmp = self.heaplst[p] 
                self.heaplst[p] = self.heaplst[i] # swap the bigger node i with its smaller parent
                self.heaplst[i] = tmp

            i = i // 2 # check now the parent of the ith node, so make ith node the parent
            p = i // 2

    def insert(self,val):
        """ @fn  insert 
            @brief insert a val at the end of the heap and then percup to check max heap property
        """
        self.heaplst.append(val)
        self.size = self.size + 1
        self.percup( self.size) # check for max heap property

    def maxChild(self,i):
        """ @fn maxChild
            @brief find the child with the maximum value for node i, i being the parent 
            @returns positive int of child's node index that has maximum value amongst children of i
        """
        l = 2*i
        r = 2*i +1
        if (r > self.size):
            return l
        else:
            if self.heaplst[l] < self.heaplst[r]:
                return r
            else:
                return l

    def percdown(self,i):
        """ @fn percdown
            @brief percolate down or "heapify"
        """
        l = 2*i # left child's index for ith node
        L = self.size
        while (l <= L):
            mc = self.maxChild(i)    
            if (self.heaplst[mc] > self.heaplst[i]): # max heap property violated; do a swap between parent and max child
                tmp = self.heaplst[i] 
                self.heaplst[i] = self.heaplst[mc]
                self.heaplst[mc] = tmp

            i = mc
            l = 2*i
    
    def delMax(self):
        """ @fn delmax 
            @brief max will always be at root because max heap property; afterwards, ensure max heap property enforced
            @returns maximum value of the heap
        """
        maxval = self.heaplst[1]
        L = self.size
        self.heaplst[1] = self.heaplst[L] # swap last, end of heap with first value, so to only do 1 swap and not move over entire thing
        self.size = L -1 # popped out 1 
        self.heaplst.pop() # pop out very last of heap
        self.percdown(1) # start from root and percolate down to enforce max heap property over entire heap

        return maxval

    def buildHeap(self,X):
        """ @fn buildHeap
            @param X - list of values for the nodes (0-based counting)
            @note Enforce the max heap property for all the parents; that means using percdown for each possible parent
        """ 
        self.heaplst = [0,] + X[:] # take the list as is
        L = len(X)
        self.size = L 
        i = L // 2 # 1st parent; L/2+1,L/2+2...are all leafs by complete binary tree structure
        while (i>0):
            self.percdown(i)
            i = i - 1

def minheapsort(X):
    sortedlst = []
    A = minHeap()
    A.buildHeap(X) 
    L=A.size
    for i in range(L,0,-1):
        sortedlst.append( A.delMin() )
    return sortedlst

def maxheapsort(X):
    sortedlst = []
    A = minHeap()
    A.buildHeap(X) 
    L=A.size
    for i in range(L,0,-1):
        sortedlst.append( A.delMax() )
    return sortedlst



"""
def minheapsort(X):
    "" @fn minheapsort
    ""
    sortedlst = []
    A = minHeap()
    A.buildHeap(X) # now A(1) is the maximum, A.size is L=|X|, length of X
    n = A.size
    while (n>1):
        minval = A.heapLst[1] 
        A.heapLst[1] = A.heapLst[n]
        A.heapLst[n] = minval

        sortedlst.append(minval)
        A.heapLst = A.heapLst[:n] # 0-based counting for Python
        n = n-1
        A.percdown(1)
    return sortedlst 


def maxheapsort(X):
    "" @fn maxheapsort
    ""
    sortedlst = []
    A = maxHeap()
    A.buildHeap(X) # now A(1) is the maximum, A.size is L=|X|, length of X
    n = A.size
    while (n>1):
        maxval = A.heapLst[1] 
        A.heapLst[1] = A.heapLst[n]
        A.heapLst[n] = maxval

        sortedlst.append(maxval)
        A.heapLst = A.heapLst[:n] # 0-based counting for Python
        n = n-1
        A.percdown(1)
    return sortedlst 
"""