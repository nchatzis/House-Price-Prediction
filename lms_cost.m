function J = lms_cost(X,y,w)
%LMS_COST Compute Least Mean Squares cost function.
%
n = height(y);
J=0;
J = (1/n)*sum((y-X*w).^2);
end

