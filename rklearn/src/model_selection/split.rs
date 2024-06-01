use crate::neural_network::base::Mtrx;

pub fn train_test_split(
    x: Mtrx<f32>, 
    y: Mtrx<f32>,
    test_size: usize,
    train_size: usize,
    random_state: usize, 
    shuffle: bool,
    stratify: Vec<f32>) -> (Mtrx<f32>, Mtrx<f32>, Mtrx<f32>, Mtrx<f32>){
        assert!(x.len() > 1 || y.len() > 1);
        
        let n_samples = x.shape().0;

        todo!()
    }