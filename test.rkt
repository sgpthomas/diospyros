#lang rosette/safe

(require threading)

(model
 (verify
  (eq? (< (+ a b) (+ c d))
       (< (+ a b) (+ c c)))))

(begin
  (clear-vc!)
  (define-symbolic a b integer?)
  ;; (define-symbolic c integer?)
  ;; (define-symbolic d integer?)
  (~>
   (verify
    (assert
     (=>
      (< a b)
      (< a (+ 1 b)))))
   )
  )


(let ([?a 66]
      [?b 89]
      [?c -6]
      [?d 83])
  (=>
   (< ?a ?b)
   (< ?a (+ 1 ?b))
   ))
