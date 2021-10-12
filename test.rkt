#lang rosette

(require threading)

(model
 (verify
  (eq? (< (+ a b) (+ c d))
       (< (+ a b) (+ c c)))))

(begin
  ;; (clear-vc!)
  ;; (define-symbolic c integer?)
  ;; (define-symbolic d integer?)
  )

(define-symbolic ?a ?b integer?)
(clear-vc!)
(vc)
(model (verify
 (begin
   (assert (< ?a ?b))
   (assert (< ?a 1))
   (assume (equal?
            (< ?a ?b)
            (< ?a 1))))))

(clear-vc!)
(vc)
(model (verify
 (begin
   (assume (< ?a ?b))
   (assert (< ?a 1))
   )))
(vc)

(parameterize ([output-smt "~/Research/diospyros/smt"])
  (result-value
   (with-vc vc-true
     (verify
      (begin
        (forall (list a b)
                (and (=> (< ?a ?b)
                         (< ?a 1))
                     (=> (< ?a 1)
                         (< ?a ?b))))
        
        )
        )

     ))
  )



(let ([?a 66]
      [?b 89]
      [?c -6]
      [?d 83])
  (=>
   (< ?a ?b)
   (< ?a (+ 1 ?b))
   ))
