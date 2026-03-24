data {

	int<lower=1> N;							// observations
	int<lower=1> S;							// sectors
	int<lower=1> K;							// severity classes
	array[N] int<lower=1,upper=S> sector;	// sector of observation
	
	vector<lower=0>[N] workers;				// workers in observation (sector-year)
	array[N,K] int<lower=0> y;				// injuries in observation by severity class
}

parameters {
	real alpha;					// global intercept
	real<lower=0> sh_sigma;
	vector[S] sh;				// sector hazard
	vector[K-1] raw_incidence;	// poisson thinning
	
	vector<lower=0>[K-1] sigma_incidence;
	array[S] vector[K-1] z_incidence;		// partially pooled incidence (std)
}

transformed parameters {
	vector[K] eta_incidence;
	eta_incidence[1] = 0.0;
	eta_incidence[2:K] = raw_incidence;

	array[S] vector[K] eta_incidence_s;
	array[S] vector[K] log_incidence;

	for (s in 1:S) {
		eta_incidence_s[s][1] = 0.0;
		eta_incidence_s[s][2:K] =
			eta_incidence[2:K] + sigma_incidence .* z_incidence[s];

		log_incidence[s] = log_softmax(eta_incidence_s[s]);
	}
}

model {

	alpha ~ normal(-4, 1);
	sh ~ normal(0, sh_sigma);
	sh_sigma ~ exponential(1);
	
	for (k in 1:(K-1)){
		raw_incidence[k] ~ normal((-1.0 * k), 1);
	}
	
	for (s in 1:S) {
		z_incidence[s] ~ std_normal();
	}
	sigma_incidence ~ exponential(1);

	for (i in 1:N) {
		real log_lambda = alpha + log(workers[i]) + sh[sector[i]];
		y[i,:] ~ poisson_log(log_lambda + log_incidence[sector[i]]);
	}
}

generated quantities {
	array[N,K] int y_rep;
	
	for (i in 1:N) {
		real log_lambda = alpha + log(workers[i]) + sh[sector[i]];
		for (k in 1:K) {
			y_rep[i,k] = poisson_log_rng(log_lambda + log_incidence[sector[i],k]);
		}
	}
}