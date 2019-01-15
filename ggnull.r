ebb_gnull=function(x,d){
	ans=array(NA,dim=c(x[1],1))
	for (j in 0:(x[1]-1)) {
		vec2 <- 0:(d-j-1)		
		vec3 <- 0:(d-1)	
		if (j==0){
			ans[j]= sum( log(1-x[2]+x[3]*vec2) ) - sum( log(1+x[3]*vec3) )
		} else {
			vec1 <- 0:(j-1)	
			ans[j]<- lchoose(d,x[1]) + sum(log(x[2]+x[3]*vec1)) + sum(log(1-x[2]+x[3]*vec2)) - sum(log(1+x[3]*vec3))
		}		
	}
	return(1-sum(exp(ans)))
}

ggnull <- function(test_stats,  pairwise_cors) {
	t_vec <- sort(abs(test_stats), decreasing=TRUE)

	d=length(t_vec)
	k_vec=1:d
	
	p_values <- 1-pchisq(t_vec^2, df=1)
	indicator <- which( p_values < k_vec/d )
	first_half <- which(k_vec <= ceiling(d/2))
	non_zero <- intersect(indicator, first_half)

	mean_null <- rep(NA, length(t_vec))
	sigsq_null <- rep(NA, length(t_vec))
	mean_null[non_zero] <- 2*d*surv(t_vec[non_zero])
	browser()
	sigsq_null <- calc_var_nonzero_mu(d=d, t=t_vec, mu=rep(0,length(t_vec)), pairwise_cors=pairwise_cors)

	mu_null <- rep(NA, length(t_vec))
	rho_null <- rep(NA, length(t_vec))
	gamma_null <- rep(NA, length(t_vec))
	mu_null[non_zero] <- mean_null[non_zero] / d
	rho_null[non_zero] <- (sigsq_null[non_zero] - d*mu_null[non_zero]*(1-mu_null[non_zero])) /
			(d*(d-1)*mu_null[non_zero]*(1-mu_null[non_zero]))
	gamma_null[non_zero] = rho_null[non_zero] / (1-rho_null[non_zero])
	
	pq_mat_null <- matrix(data=NA, nrow=length(t_vec), ncol=2)
	pq_mat_null[non_zero, 1] <- -mu_null[non_zero] / (d-1)
	pq_mat_null[non_zero, 2] <- -(1-mu_null[non_zero]) / (d-1)
	gamma_check_null <- apply(pq_mat_null, 1, max)
		
	non_zero <- which(gamma_null >= gamma_check_null)
	return(sigsq_null)
	if (length(non_zero)==0 ) {
		return ( NA)
	}

	null_param_mat <- cbind(k_vec[non_zero],mu_null[non_zero],gamma_null[non_zero])
	stats <- apply(null_param_mat, 1, ebb_gnull, d=d)

	return(max(-stats))
}

#ggnull(rep(0,100),rep(0,50*99))