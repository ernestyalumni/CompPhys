# $SO(3)$ Lie Group implementation using GNU GSL
`SO3.c`
=======

I want to implement the following generators, a basis for the (vector space, by definition of a Lie algebra) Lie algebra of $SO(3)$ in this representation for $\mathfrak{so}(3)$:

\[
\begin{gathered}
	L_x = \left[ \begin{matrix}
	    0 & 0 & 0
	    0 & 0 & -1 \\
	    0 & 1 & 0 \end{matrix} \right] \qquad \, 	L_y = \left[ \begin{matrix}
	    0 & 0 & 1
	    0 & 0 & 0 \\
	    -1 & 0 & 0 \end{matrix} \right] \qquad \, 
	L_z = \left[ \begin{matrix}
	    0 & -1 & 0
	    1 & 0 & 0 \\
	    0 & 0 & 0 \end{matrix} \right] \qquad \, 
\end{gathered}
\]