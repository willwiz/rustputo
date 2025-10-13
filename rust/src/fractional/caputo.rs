pub mod caputo_internal;
pub mod caputo_ndarray;
#[cfg(test)]
mod tests {
    use super::caputo_internal::CaputoInternal;
    use crate::fractional::derivatives::LinearDerivative;
    use std::array;

    #[test]
    fn b0_interpolation() {
        let val = CaputoInternal::<1, 9>::new(0.1, 0.0, 1.0);
        assert!((val.caputo.b0 - 1.695090131733033e-05).abs() < 1e-15);
    }

    #[test]
    fn beta_interpolation() {
        let val = CaputoInternal::<1, 9>::new(0.1, 0.0, 1.0);
        let benchmark = [
            0.3807318762367083,
            0.22845158393131948,
            0.20291513377934745,
            0.19369053648893741,
            0.18765902406556312,
            0.1835429051341629,
            0.18325462110394497,
            0.21370479375327925,
            0.7562112906391003,
        ];
        for (a, b) in benchmark.iter().zip(val.caputo.beta.iter()) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }

    #[test]
    fn tau_interpolation() {
        let val = CaputoInternal::<1, 9>::new(0.1, 0.0, 1.0);
        let benchmark = [
            0.0001947414850719321,
            0.0007090855132484599,
            0.002144537588884077,
            0.006785451628166409,
            0.023698885971713417,
            0.09501887557390602,
            0.4656170456619207,
            3.388094041697312,
            50720.792285294534,
        ];
        println!("tau: {:?}", val.caputo.tau);
        println!("benchmark: {:?}", benchmark);
        for (a, b) in benchmark.iter().zip(val.caputo.tau.iter()) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }

    #[test]
    fn b0_interpolation2() {
        let val = CaputoInternal::<1, 9>::new(0.166, 0.0, 1.0);
        assert!((val.caputo.b0 - 5.1249158338074934e-05).abs() < 1e-15);
    }

    #[test]
    fn beta_interpolation2() {
        let val = CaputoInternal::<1, 9>::new(0.166, 0.0, 1.0);
        let benchmark = [
            1.116177194866666,
            0.5946435683254347,
            0.4880965776074402,
            0.4344171743437531,
            0.3913707103773119,
            0.353504390222247,
            0.322511690639043,
            0.32494993367312547,
            0.639194618020289,
        ];
        for (a, b) in benchmark.iter().zip(val.caputo.beta.iter()) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }

    #[test]
    fn tau_interpolation2() {
        let val = CaputoInternal::<1, 9>::new(0.166, 0.0, 1.0);
        let benchmark = [
            0.00017679049886707728,
            0.0006430988523212297,
            0.0018963341950322585,
            0.005822947295416858,
            0.019772488823216,
            0.07756643518358002,
            0.3773468555358873,
            2.76978619070747,
            50720.792285294534,
        ];
        println!("tau: {:?}", val.caputo.tau);
        println!("benchmark: {:?}", benchmark);
        for (a, b) in benchmark.iter().zip(val.caputo.tau.iter()) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }

    #[test]
    fn caputo_works() {
        let mut caputo_init = CaputoInternal::<1, 9>::new(0.1, 0.0, 1.0);
        let fvals = [[0.0], [1.0], [2.0], [3.0]];
        let dfvals: [[f64; 1]; 4] =
            array::from_fn(|i| caputo_init.caputo_derivative_lin(&fvals[i], 0.1));
        let benchmark = [
            0.0,
            1.258940005546393,
            2.3923038940303556,
            3.469186999298201,
        ];
        // [0.0, 1.258940005546393, 2.3923038940303556, 3.469186999298201, ] Without dt term
        // [0.0, 1.2591095145595663, 2.392473403043529, 3.4693565083113747, ] With dt term
        print!("dfvals: {:?}\n", dfvals);
        print!("benchmark: {:?}\n", benchmark);
        let check = benchmark
            .iter()
            .zip(dfvals.iter())
            .all(|(a, b)| (a - b[0]).abs() < 1e-14);
        assert_eq!(check, true);
    }
}
