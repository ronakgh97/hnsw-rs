fn main() {
    let cat = "cat";
    let dog = "dog";
    assert_eq!(levenshtein_distance(cat, dog), 3);
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev: Vec<usize> = (0..=n).collect(); // row 0
    let mut curr = vec![0usize; n + 1]; // row 1

    for i in 1..=m {
        curr[0] = i; // first column of current row
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1) // delete
                .min(curr[j - 1] + 1) // insert
                .min(prev[j - 1] + cost); // substitute
        }
        std::mem::swap(&mut prev, &mut curr); // move to next row
    }

    prev[n] // result after last swap puts final row in `prev`
}

#[test]
fn test_levenshtein_distance() {
    assert_eq!(levenshtein_distance("", ""), 0);
    assert_eq!(levenshtein_distance("a", ""), 1);
    assert_eq!(levenshtein_distance("ab", "a"), 1);
    assert_eq!(levenshtein_distance("ab", "ab"), 0);
}
