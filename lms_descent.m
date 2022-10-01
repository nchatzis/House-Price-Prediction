function w = lms_descent(X,y,w,r,iter)
%LMS_DESCENT Gradient descent to find w.
% Update w for iter iterations or stop if cost function converges.
n = length(y);

for i=1:iter
    J_prev = lms_cost(X,y,w);   %find previous cost
    
    w = w - 2*(r/n) * X' * (X*w - y);
    
    J = lms_cost(X,y,w); %find new cost
    
    if J-J_prev<0.0001  %if convergence
        break;
    end
    
end

end

