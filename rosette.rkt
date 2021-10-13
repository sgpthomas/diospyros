#lang rosette/safe

(define-symbolic a b integer?)

(model
 (result-value
 (with-vc vc-true
  (begin
    (solve
     (assert
      (not (eq? (< a 0)
                (< a b)))))
    )
  )))

(with-vc vc-true
  (begin
    (solve
     ()
     )
    )
  )

(clear-vc!)
(vc)

(assert (forall ((x Int)
                 (y Int))
                (=> (x < y)
                    (x < y))))
