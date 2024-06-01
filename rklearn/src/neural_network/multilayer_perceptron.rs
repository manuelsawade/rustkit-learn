use std::iter::zip;

use super::base::Activation;
use super::base::Derivatives;
use super::base::LossFunction;
use super::base::Mtrx;
use super::stochastic_optimizers::*;

enum MLPType{
    Classifier,
    Regressor
}

#[derive(PartialEq, Clone, Copy)]
pub enum LearningRate{
    Constant,
    Invscaling,
    Adaptive
}

struct BaseMultilayerPerceptron{
    hidden_layer_sizes: &'static [usize],
    activation: Activation,
    solver: StochasticSolver,
    alpha: f32,
    batch_size: usize,
    learning_rate: LearningRate, 
    learning_rate_init: f32,
    power_t: f32,
    max_iter: usize,
    loss: LossFunction,
    shuffle: bool,
    random_state: usize,
    tol: f32,
    verbose: bool,
    warm_start: bool,
    momentum: f32,
    nesterovs_momentum: bool,
    early_stopping: bool,
    validation_fraction: usize,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    n_iter_no_change: usize,
    max_fun: usize,
    //internal
    n_outputs: usize,
    n_iter: usize,
    t: usize,
    n_layers: usize,
    out_activation: Activation,
    coefs: Mtrx<f32>,
    intercepts: Vec<f32>,
    loss_curve: Vec<f32>,
    no_improvement_count: usize,
    validation_scores: Option<Vec<f32>>,
    best_validation_scores: Option<f32>,
    best_loss: Option<f32>
}

impl BaseMultilayerPerceptron{
    pub fn init(
        hidden_layer_sizes: &'static [usize],
        activation: Activation,
        solver: StochasticSolver,
        alpha: f32,
        batch_size: usize,
        learning_rate: LearningRate, 
        learning_rate_init: f32,
        power_t: f32,
        max_iter: usize,
        loss: LossFunction,
        shuffle: bool,
        random_state: usize,
        tol: f32,
        verbose: bool,
        warm_start: bool,
        momentum: f32,
        nesterovs_momentum: bool,
        early_stopping: bool,
        validation_fraction: usize,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        n_iter_no_change: usize,
        max_fun: usize,
        mlp_type: MLPType
    ) -> Self{
        BaseMultilayerPerceptron{
            hidden_layer_sizes,
            activation,
            solver,
            alpha,
            batch_size,
            learning_rate, 
            learning_rate_init,
            power_t,
            max_iter,
            loss,
            shuffle,
            random_state,
            tol,
            verbose,
            warm_start,
            momentum,
            nesterovs_momentum,
            early_stopping,
            validation_fraction,
            beta_1,
            beta_2,
            epsilon,
            n_iter_no_change,
            max_fun,
            n_outputs: usize::default(),
            n_iter: usize::default(),
            t: usize::default(),
            n_layers: 0,
            out_activation: match mlp_type{
                MLPType::Regressor => Activation::Identity,
                MLPType::Classifier => Activation::Logistic
            },
            coefs: Mtrx::new(Vec::new()),
            intercepts: vec![0f32],
            loss_curve: vec![0f32],
            no_improvement_count: usize::default(),
            validation_scores: None,
            best_validation_scores: None,
            best_loss: None
        }
    }

    fn fit(&mut self, x: &mut Mtrx<f32>, y: &mut Mtrx<f32>, incremental: bool){
        assert!(x.len() >= 1usize, "No samples given in x.");
        assert!(x[0].len() >= 1usize, "No features given in x.");

        let first_pass = !self.warm_start && !incremental;

        let n_features = x.shape().1;
        self.n_outputs = y.shape().1;

        let mut layer_units = vec![n_features];
        layer_units.append(&mut self.hidden_layer_sizes.to_vec());
        layer_units.append(&mut vec![self.n_outputs]);

        if first_pass {
            self.initialize(y, &layer_units);
        }

        let mut coef_grad = Vec::new();

        let z1 = layer_units[0..layer_units.len()-2].to_vec();
        let z2 = layer_units[1..].to_vec();

        for (n_fan_in, n_fan_out) in zip(z1, z2){
            coef_grad.push(Mtrx::from((n_fan_in, n_fan_out)))
        }

    }

    fn fit_stochastic(&mut self, 
        x: &Mtrx<f32>, 
        y: &Mtrx<f32>, 
        activations: Vec<f32>, 
        deltas: Vec<f32>,
        coef_grads: Vec<Mtrx<f32>>,
        intercept_grads: Vec<Mtrx<f32>>,
        layer_units: &Vec<usize>,
        incremental: bool
    ){
        let params = Vec::new();
        let 
        if !incremental {           
            let optimizer = self.build_stochastic_optimizer(params);


            
        };
    }
    
    fn build_stochastic_optimizer(&self, params: Vec<f32>) -> Box<dyn StochasticOptimizer>{
        match self.solver{
            StochasticSolver::SGD => Box::new(SGDOptimizer::init(
                params, 
                self.learning_rate_init, 
                self.learning_rate, 
                self.momentum, 
                self.nesterovs_momentum, 
                self.power_t))
            ,       
            StochasticSolver::Adam => Box::new(AdamOptimizer::init(
                params, 
                self.learning_rate_init, 
                self.beta_1, 
                self.beta_2, 
                self.epsilon))
        }
    }

    fn initialize(&mut self, y: &Mtrx<f32>, layer_units: &Vec<usize>){
        self.n_iter = 0;
        self.t = 0;
        self.n_outputs = y.shape().1;
        self.n_layers = layer_units.len();

        for i in 0..self.n_layers-1{
            let (mut coef_init, intercept) = self.initialize_coefs(layer_units[i], layer_units[i+1]);

            self.coefs.src_mut().append(coef_init.src_mut());
            self.intercepts.push(intercept);
        }

        if self.early_stopping {
            self.validation_scores = Some(vec![0f32]);
            self.best_validation_scores = Some(f32::NEG_INFINITY);
        } else {
            self.best_loss = Some(f32::INFINITY);
        }
    }

    fn initialize_coefs(&mut self, fan_in: usize, fan_out: usize) -> (Mtrx<f32>, f32){
        use super::multilayer_perceptron::*;
        
        let factor = 
            if let Activation::Logistic = self.activation {2f32} 
            else {6f32};
        let init_bound = (factor / (fan_in as f32 + fan_out as f32) ).sqrt();

        let coef_init = uniform_shape(
            -init_bound, init_bound, (fan_in, fan_out));

        let intercept_init = uniform_single(-init_bound, init_bound, fan_out);   
        (coef_init, intercept_init)
    }
}

fn uniform_single(low: f32, high: f32, size: usize) -> f32{
    todo!()
}

fn uniform_shape(low: f32, high: f32, size: (usize, usize)) -> Mtrx<f32>{
    todo!()
}