
from numpy import size, zeros, transpose, ones, logical_or, array, copy, eye, vstack, tile, diag, cumsum, empty, prod
def poldif(x, malpha=0, B=0):
    """
    function DM = poldif(x, malpha, B)
    
      The function DM =  poldif(x, maplha, B) computes the
      differentiation matrices D1, D2, ..., DM on arbitrary nodes.
    
      The function is called with either two or three input arguments.
      If two input arguments are supplied, the weight function is assumed 
      to be constant.   If three arguments are supplied, the weights should 
      be defined as the second and third arguments.
    
      Input (constant weight):
    
      x:        Vector of N distinct nodes.
      malpha:   M, the number of derivatives required (integer).
      B:        Omitted.
    
      Note:     0 < M < N-1.
    
      Input (non-constant weight):
    
      x:        Vector of N distinct nodes.
      malpha:   Vector of weight values alpha(x), evaluated at x = x(k).
      B:        Matrix of size M x N,  where M is the highest 
                derivative required.  It should contain the quantities 
                B(ell,j) = beta(ell,j) = (ell-th derivative
                of alpha(x))/alpha(x),   evaluated at x = x(j).
    
      Output:
      DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
    
      J.A.C. Weideman, S.C. Reddy 1998
    """
    N= size(x)
    if malpha != 0:                     # Check if constant weight function
        M = malpha                      # is to be assumed.
        alpha = ones(N)              
        B = zeros((M,N))
        
    elif B != 0:
        alpha = malpha                  # Make sure alpha is a column vector
        M = size(B[:,1],1)              # First dimension of B is the number 
    I = eye(N)                       # Identity matrix.
    L = logical_or(I,zeros(N))    # Logical identity matrix. 
    XX = transpose(array([x,]*N))
    DX = XX-transpose(XX)            # DX contains entries x(k)-x(j). 
    DX[L] = ones(N)                  # Put 1's one the main diagonal. 
    c = alpha*prod(DX,1)             # Quantities c(j). 
    C = transpose(array([c,]*N)) 
    C = C/transpose(C)               # Matrix with entries c(k)/c(j).    
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                  # with zeros on the diagonal.      
    X = transpose(copy(Z))                 # X is same as Z', but with 
    Xnew=X                
    for i in range(0,N):
        Xnew[i:N-1,i]=X[i+1:N,i]
    X=Xnew[0:N-1,:]                  # diagonal entries removed. 
    Y = ones([N-1,N])                    # Initialize Y and D matrices.
    D = eye(N);                         # Y is matrix of cumulative sums,
    DM=empty((N,N,M))                                   # D differentiation matrices.
    for ell in range(1,M+1):
        Y=cumsum(vstack((B[ell-1,:], ell*(Y[0:N-1,:])*X)),0) # Diagonals
        D=ell*Z*(C*transpose(tile(diag(D),(N,1))) - D)   # Off-diagonal         
        D[L]=Y[N-1,:]
        DM[:,:,ell-1] = D 
    return DM
#D=poldif(linspace(0,1,4),2)
#print D[:,:,1] 