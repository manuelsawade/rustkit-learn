use std::{ops::{Deref, DerefMut}, iter::zip};

use super::multilayer_perceptron::LearningRate;

pub struct BaseOptimizer{
    learning_rate_init: f32,
    learning_rate: f32,  
}

pub trait StochasticOptimizer{
    fn update_params(&mut self, grads: &Vec<f32>) -> Vec<f32>;
}

impl BaseOptimizer{
    pub fn init(learning_rate_init: f32) -> Self{
        BaseOptimizer { 
            learning_rate_init,
            learning_rate: 0f32
        }
    }
}

pub struct SGDOptimizer{
    base: BaseOptimizer,
    params: Vec<f32>,
    lr_schedule: LearningRate,
    momentum: f32,
    nesterov: bool,
    power_t: f32,
    velocities: Vec<f32>
}

impl<'a> Deref for SGDOptimizer{
    type Target = BaseOptimizer;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'a> DerefMut for SGDOptimizer{
    fn deref_mut(&mut self) -> &mut BaseOptimizer {
        &mut self.base
    }
}

impl SGDOptimizer{
    pub fn init(
        params: Vec<f32>, 
        learning_rate_init: f32,
        lr_schedule: LearningRate,
        momentum: f32,
        nesterov: bool,
        power_t: f32
    ) -> Self {
        let params_len = params.len();
        SGDOptimizer{
            base: BaseOptimizer::init(learning_rate_init),
            params,
            lr_schedule,
            momentum,
            nesterov,
            power_t,
            velocities: vec![0f32; params_len]
        }
    }

    pub fn iteration_ends(&mut self, timestep: usize){
        if let LearningRate::Invscaling = self.lr_schedule {
            self.learning_rate = 
                self.learning_rate_init / 
                (timestep as f32 + 1f32).powf(self.power_t)
        }
    }

    pub fn trigger_stopping(&mut self, msg: String, verbose: bool) -> bool{
        if self.lr_schedule != LearningRate::Adaptive{
            if verbose { print!("{msg} Stopping"); }
            return true;
        }

        if self.learning_rate <= 1e-6 {
            if verbose { print!("{msg} Learning rate too small. Stopping."); }
            return true;
        }

        let new_rate = 5f32;
        self.learning_rate /= new_rate;
        if verbose { print!("{msg}  Setting learning rate to {new_rate}"); }
        return false;
    }
}

impl StochasticOptimizer for SGDOptimizer{
    fn update_params(&mut self, grads: &Vec<f32>) -> Vec<f32> {
        let mut updates = vec![0f32; grads.len()];
        for (i, (velocity, grad)) in zip(&self.velocities, grads).enumerate(){
            updates[i] = self.momentum * velocity - self.learning_rate * grad;
        }

        self.velocities = updates.clone();

        if self.nesterov {
            for (i, (velocity, grad)) in zip(&self.velocities, grads).enumerate(){
                updates[i] = self.momentum * velocity - self.learning_rate * grad;
            }
        }
        
        updates
    }
}

pub struct AdamOptimizer{
    base: BaseOptimizer,
    params: Vec<f32>,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    t: f32,
    ms: Vec<f32>,
    vs: Vec<f32>
}

impl<'a> Deref for AdamOptimizer{
    type Target = BaseOptimizer;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'a> DerefMut for AdamOptimizer{
    fn deref_mut(&mut self) -> &mut BaseOptimizer {
        &mut self.base
    }
}

impl AdamOptimizer{
    pub fn init(
        params: Vec<f32>,
        learning_rate_init: f32,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32
    ) -> Self{
        let params_len = params.len();
        AdamOptimizer{
            base: BaseOptimizer::init(learning_rate_init),
            params,
            beta_1,
            beta_2,
            epsilon,
            t: 0f32,
            ms: vec![0f32; params_len],
            vs: vec![0f32; params_len]
        }
    }
}

impl StochasticOptimizer for AdamOptimizer{
    fn update_params(&mut self, grads: &Vec<f32>) -> Vec<f32> {
        self.t += 1f32;
        self.ms = zip(&self.ms, grads).map(|(m,grad)| {
                self.beta_1 * m + (1f32 - self.beta_1) * grad
            }).collect();

        self.vs = zip(&self.vs, grads).map(|(v,grad)| {
                self.beta_2 * v + (1f32 - self.beta_2) * (grad.powf(2f32))
            }).collect();

        self.learning_rate = 
            self.learning_rate_init * 
            (1f32 - self.beta_2.powf(self.t)).sqrt() /
            (1f32 - self.beta_1.powf(self.t));

        zip(&self.ms, &self.vs).map(|(m,v)| {
            -self.learning_rate * m / (v.sqrt() + self.epsilon)
        }).collect()
    }
}

pub enum StochasticSolver{
    SGD,
    Adam
}