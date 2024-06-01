use std::{f32::consts::E, iter::zip, ops::{Deref, DerefMut}};

pub struct Mtrx<Number>{
    source: Vec<Vec<Number>>
}

impl Mtrx<f32>{
    pub fn shape(&self) -> (usize, usize){
        (self.len(), self[0].len())
    }

    pub fn src_mut(&mut self) -> &mut Vec<Vec<f32>>{
        &mut self.source
    }

    pub fn src(& self) -> &Vec<Vec<f32>>{
        &self.source
    }

    pub fn new(vec: Vec<Vec<f32>>) -> Self{
        Mtrx { source: vec }
    }
    
    pub fn from(shape: (usize, usize)) -> Self{
        Mtrx { source: vec![vec![0f32; shape.1]; shape.0]}
    }
}

impl Deref for Mtrx<f32>{
    type Target = Vec<Vec<f32>>;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl DerefMut for Mtrx<f32>{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.source
    }
}

///Compute the logistic function inplace.
///# Arguments
///* `x` - The input data.
fn logistic(x: &mut Mtrx<f32>){   
    for feat in x.iter_mut().flatten() {
        *feat = 1f32 / 1f32 + E.powf(-*feat);              
    };
}

///Compute the hyperbolic tan function inplace.
///# Arguments
///* `x` - The input data.
fn tanh(x: &mut Mtrx<f32>){   
    for feat in x.iter_mut().flatten() {
            *feat = feat.tanh();       
    };
}

///Compute the rectified linear unit function inplace.
///# Arguments
///* `x` - The input data.
fn relu(x: &mut Mtrx<f32>){   
    for feat in x.iter_mut().flatten() {
        *feat = if *feat <= 0f32 { 0f32 } else { 1f32 };             
    };
}

///Compute the K-way softmax function inplace.
///# Arguments
///* `x` - The input data.
fn softmax(x: &mut Mtrx<f32>){   
    for samp in x.iter_mut() {
        let sum_exp = samp.iter()
            .map(|s| E.powf(*s)).sum::<f32>();
        
        for feat in samp.iter_mut() {
            *feat = *feat / sum_exp;
        };       
    };
}

pub enum Activation{
    Identity,
    Tanh,
    Logistic,
    Relu,
    Softmax
}

impl Activation{    
    pub fn func(self, x: &mut Mtrx<f32>){
        match self{
            Self::Tanh => tanh(x),
            Self::Logistic => logistic(x),
            Self::Relu => relu(x),
            Self::Softmax => softmax(x),
            Self::Identity => ()
        };
    }
}

///Apply the derivative of the logistic sigmoid function.
///It exploits the fact that the derivative is a simple function of the output
///value from logistic function.
///# Arguments
///* `z` - The data which was output from the logistic activation 
///        function during the forward pass.
///* `delta` - The backpropagated error signal to be modified inplace.
fn logistic_derivative(z: &Mtrx<f32>, delta: &mut Mtrx<f32>){
    for (z, delta) in zip(z.src(), delta.src_mut()){
        for (z, delta) in zip(z, delta){
            *delta *= *z;
            *delta *= 1f32 - *z;
        }
    }
}

///Apply the derivative of the hyperbolic tanh function.
///It exploits the fact that the derivative is a simple function of the output
///value from hyperbolic tangent.
///# Arguments
///* `z` - The data which was output from the hyperbolic tangent activation 
///        function during the forward pass.
///* `delta` - The backpropagated error signal to be modified inplace.
fn tanh_derivate(z: &Mtrx<f32>, delta: &mut Mtrx<f32>){
    for (z, delta) in zip(z.src(), delta.src_mut()){
        for (z, delta) in zip(z, delta){
            *delta *= 1f32 - z.powf(2f32);
        }
    }
}

///Apply the derivative of the relu function.
///It exploits the fact that the derivative is a simple function of the output
///value from rectified linear units activation function.
///# Arguments
///* `z` - The data which was output from the rectified linear units activation
///        function during the forward pass.
///* `delta` - The backpropagated error signal to be modified inplace.
fn relu_derivative(z: &Mtrx<f32>, delta: &mut Mtrx<f32>){
    for (z, delta) in zip(z.src(), delta.src_mut()){
        for (z, mut delta) in zip(z, delta){
            *delta = if *z == 0f32 { 0f32 } else { *delta };
        }
    }
}

pub enum Derivatives{
    Identity,
    Tanh,
    Logistic,
    Relu
}

impl Derivatives{
    pub fn func(self, z: &Mtrx<f32>, delta: &mut Mtrx<f32>){     
        match self{
            Self::Tanh => tanh_derivate(z, delta),
            Self::Logistic => logistic_derivative(z, delta),
            Self::Relu => relu_derivative(z, delta),
            Self::Identity => ()
        };
    }
}

fn squared_loss(y_true: &Mtrx<f32>, y_pred: &Mtrx<f32>) -> f32{
    let mut err = vec![0f32; y_true.len()];
    for (y_true, y_pred) in zip(&**y_true, &**y_pred){
        for (i,(y_true, y_pred)) in zip(y_true, y_pred).enumerate(){
            err[i] = (y_true - y_pred).powf(2f32);
        }
    };

    (err.iter().sum::<f32>() / err.len() as f32) / 2f32 
}

fn log_loss(y_true: &Mtrx<f32>, y_pred: &Mtrx<f32>) -> f32{
    let mut log = vec![0f32; y_true.len()];
    for (y_true, y_pred) in zip(y_true.src(), y_pred.src()){
        for (i,(y_true, y_pred)) in zip(y_true, y_pred).enumerate(){
            log[i] = -y_true * y_pred.log(E);
        }
    };

    log.iter().sum::<f32>() / y_pred.len() as f32
}

fn binary_log_loss(y_true: &Mtrx<f32>, y_pred: &Mtrx<f32>) -> f32{
    let mut xLogY1 = vec![0f32; y_true.len()];
    let mut xLogY2 = vec![0f32; y_true.len()];

    for (y_true, y_pred) in zip(y_true.src(), y_pred.src()){
        for (i,(y_true, y_pred)) in zip(y_true, y_pred).enumerate(){
            xLogY1[i] = y_true * y_pred.log(E);
            xLogY2[i] = (1f32 - y_true) * (1f32 - y_pred).log(E);
        }
    };

    -(xLogY1.iter().sum::<f32>() + xLogY2.iter().sum::<f32>()) / 
    y_pred.len() as f32
}

pub enum LossFunction{
    SquaredError,
    LogLoss,
    BinaryLogLoss
}

impl LossFunction{
    pub fn func(self, y_true: &Mtrx<f32>, y_pred: &Mtrx<f32>) -> f32{
        match self{
            Self::SquaredError => squared_loss(y_true, y_pred),
            Self::LogLoss => log_loss(y_true, y_pred),
            Self::BinaryLogLoss => binary_log_loss(y_true, y_pred)
        }
    }
}