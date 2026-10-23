//! Just enough dense linear algebra for 3-5 dimensional matrices.

pub type Mat<const R: usize, const C: usize> = [[f64; C]; R];

pub fn eye<const N: usize>() -> Mat<N, N> {
    let mut m = [[0.0; N]; N];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

pub fn mul<const R: usize, const K: usize, const C: usize>(a: &Mat<R, K>, b: &Mat<K, C>) -> Mat<R, C> {
    let mut out = [[0.0; C]; R];
    for i in 0..R {
        for k in 0..K {
            let aik = a[i][k];
            for j in 0..C {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

pub fn mul_vec<const R: usize, const C: usize>(a: &Mat<R, C>, x: &[f64; C]) -> [f64; R] {
    let mut out = [0.0; R];
    for i in 0..R {
        for j in 0..C {
            out[i] += a[i][j] * x[j];
        }
    }
    out
}

pub fn transpose<const R: usize, const C: usize>(a: &Mat<R, C>) -> Mat<C, R> {
    let mut out = [[0.0; R]; C];
    for i in 0..R {
        for j in 0..C {
            out[j][i] = a[i][j];
        }
    }
    out
}

pub fn add<const R: usize, const C: usize>(a: &Mat<R, C>, b: &Mat<R, C>) -> Mat<R, C> {
    let mut out = *a;
    for i in 0..R {
        for j in 0..C {
            out[i][j] += b[i][j];
        }
    }
    out
}

pub fn sub<const R: usize, const C: usize>(a: &Mat<R, C>, b: &Mat<R, C>) -> Mat<R, C> {
    let mut out = *a;
    for i in 0..R {
        for j in 0..C {
            out[i][j] -= b[i][j];
        }
    }
    out
}

pub fn scale<const R: usize, const C: usize>(a: &Mat<R, C>, s: f64) -> Mat<R, C> {
    let mut out = *a;
    for row in &mut out {
        for v in row {
            *v *= s;
        }
    }
    out
}

/// Gauss-Jordan with partial pivoting. None if (numerically) singular.
pub fn inv<const N: usize>(a: &Mat<N, N>) -> Option<Mat<N, N>> {
    let mut m = *a;
    let mut out = eye::<N>();
    for c in 0..N {
        let p = (c..N).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, p);
        out.swap(c, p);
        let d = m[c][c];
        for j in 0..N {
            m[c][j] /= d;
            out[c][j] /= d;
        }
        for i in 0..N {
            if i != c {
                let f = m[i][c];
                if f != 0.0 {
                    for j in 0..N {
                        m[i][j] -= f * m[c][j];
                        out[i][j] -= f * out[c][j];
                    }
                }
            }
        }
    }
    Some(out)
}

/// Matrix exponential by scaling and squaring (small, well-behaved matrices).
pub fn expm<const N: usize>(a: &Mat<N, N>) -> Mat<N, N> {
    let norm = a.iter().map(|row| row.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    let s = if norm > 0.25 { (norm / 0.25).log2().ceil() as i32 } else { 0 };
    let a = scale(a, 1.0 / 2f64.powi(s));
    let mut out = eye::<N>();
    let mut term = eye::<N>();
    for k in 1..12 {
        term = scale(&mul(&term, &a), 1.0 / k as f64);
        out = add(&out, &term);
    }
    for _ in 0..s {
        out = mul(&out, &out);
    }
    out
}
