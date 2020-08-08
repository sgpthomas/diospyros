#lang rosette
(require "../ast.rkt"
         "../configuration.rkt"
         "../utils.rkt"
         "matrix-multiply.rkt")

(provide qr-decomp:only-spec
         qr-decomp:keys)

;; Runs the spec with symbolic inputs and returns:
;; - the resulting formula.
;; - the prelude instructions (list)
;; - outputs that the postlude should write to
(define (qr-decomp:only-spec config)
  (define n (hash-ref config 'N))
  (define A (make-symbolic-matrix n n 'A))

  (define prelude
    (list
      (vec-extern-decl 'A (* n n) input-tag)
      (vec-extern-decl 'Q (* n n) output-tag)
      (vec-extern-decl 'R (* n n) output-tag)
      (vec-const 'Z (make-bv-list-zeros 1) float-type)))

  (define-values (Q R) (househoulder A))

  ; (values (list (matrix-elements Q) (matrix-elements R))
  (values (matrix-elements Q)
          (prog prelude)
          ; (list 'Q 'R)))
           (list 'Q)))

(define qr-decomp:keys
  (list 'N 'reg-size))

(define (matrix-transpose m)
  (match-define (matrix rows cols elems) m)
  (define out (matrix cols rows (make-bv-list-zeros (* rows cols))))
  (for* ([i rows]
         [j cols])
    (matrix-set! out j i (matrix-ref m i j)))
  out)

(define (eq-as-value? i j)
  (if (equal? i j) (bv-value 1) (bv-value 0)))

(define-symbolic bv-sqrt (~> (bitvector (value-fin))
                          (bitvector (value-fin))))
(define-symbolic bv-sgn (~> (bitvector (value-fin))
                          (bitvector (value-fin))))
(define (sqrt-mock x)
  (bv-value (round (sqrt (bitvector->integer x)))))

(define (vector-norm v
                     [sqrt-func bv-sqrt])
                    ; [sqrt-func sqrt-mock])
  (define (bv-sqr x) (bvmul (unbox x) (unbox x)))
  (sqrt-func (apply bvadd (map bv-sqr v))))

(define (q-i q-min i j k)
  (if (or (< i k) (< j k))
    (eq-as-value? i j)
    (matrix-ref q-min (- i k) (- j k))))

(define (identity n)
  (define I (matrix n n (make-bv-list-zeros (* n n))))
  (for* ([i (in-range n)]
         [j (in-range n)])
    (matrix-set! I i j (eq-as-value? i j)))
  I)

(define (sgn-mock x)
  (cond
    [(bvslt x (bv-value 0)) (bv-value -1)]
    [(bvsgt x (bv-value 0)) (bv-value 1)]
    [else (bv-value 0)]))

(define (househoulder A
                     [sgn-func bv-sgn])
                     ; [sgn-func sgn-mock])
  (match-define (matrix A-rows A-cols A-elements) A)
  (assert (equal? A-rows A-cols))
  (define n A-rows)

  ; Initialize R to be a copy of A's elements
  (define R (matrix n n (map box (map unbox A-elements))))

  ; Create Q as a zero matrix of the same size
  (define Q (matrix n n (make-bv-list-zeros (* n n))))

  ; Create identity of the same size
  (define I (identity n))

  (for ([k (in-range (sub1 n))])

    ; Create the vectors x, e (length dependent on n minus index, call this m)
    (define m (- n k))
    (define x (make-bv-list-zeros m))
    (define e (make-bv-list-zeros m))
    (for ([row (in-range k n)]
          [i (in-naturals 0)])
     (bv-list-set! x (bv-index i) (matrix-ref R row k))
     (bv-list-set! e (bv-index i) (matrix-ref I row k)))

    ; alpha is a scalar
    (define alpha
      (bvmul (bvsub (sgn-func (bv-list-get x (bv-index 0))))
                    (vector-norm x)))

    ; u and v length based on x and e
    (define u (make-bv-list-zeros m))
    (define v (make-bv-list-zeros m))

    ; Calculate u
    (for ([i (in-range m)])
      (define u-i (bvadd (bv-list-get x (bv-index i))
                         (bvmul alpha
                                (bv-list-get e (bv-index i)))))
      (bv-list-set! u (bv-index i) u-i))

    ; Calculate v
    (define norm-u (vector-norm u))
    (for ([i (in-range m)])
      (define v-i (bvsdiv (bv-list-get u (bv-index i))
                          norm-u))
      (bv-list-set! v (bv-index i) v-i))

    ; Create the Q minor matrix
    (define Q-min (matrix m m (make-bv-list-zeros (* m m))))
    (for* ([i (in-range m)]
           [j (in-range m)])
      (define q-min-i
        (bvsub (eq-as-value? i j)
               (bvmul (bv-value 2)
                      (bv-list-get v (bv-index i))
                      (bv-list-get v (bv-index j)))))
       (matrix-set! Q-min i j q-min-i))

    ; "Pad out" the Q minor matrix with elements from the identity
    (define Q-t (matrix n n (make-bv-list-zeros (* n n))))
    (for* ([i (in-range n)]
           [j (in-range n)])
       (matrix-set! Q-t i j (q-i Q-min i j k)))

    (if (equal? k 0)
      (begin
        (set-matrix-elements! Q
                              (matrix-elements Q-t))
        (set-matrix-elements! R
                              (matrix-multiply-spec Q-t A)))
      ; Else not first iteration
      (begin
        (set-matrix-elements! Q
                              (matrix-multiply-spec Q-t Q))
        (set-matrix-elements! R
                              (matrix-multiply-spec Q-t R)))))

  (values (matrix-transpose Q) R))


;(define in (make-symbolic-matrix 3 3))
; [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
; (define in (matrix 3
;                    3
;                    (value-bv-list 12 -51 4 6 167 -68 -4 24 -41)))
; (pretty-print in)
; (pretty-print (matrix-transpose in))

; (define-values (Q R) (househoulder in))
; (pretty-print (map bitvector->integer (map unbox (matrix-elements Q))))
; (pretty-print (map bitvector->integer (map unbox (matrix-elements R))))

; (pretty-print 'done)
