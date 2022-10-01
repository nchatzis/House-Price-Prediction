function w = least_squares(X,y)
%LEAST_SQUARES Find weight vector using least squares method. 

w = inv(X'*X)*X'*y;

end

