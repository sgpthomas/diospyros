#lang rosette

(require "../ast.rkt"
         "../dsp-insts.rkt"
         "../interp.rkt"
         "../utils.rkt"
         "../prog-sketch.rkt"
         "../synth.rkt"
         "../configuration.rkt"
         racket/trace
         racket/generator
         rosette/lib/angelic)

;(provide dot-product:keys
;         dot-product:run-experiment)

;; Generate a spec for dot product between two lists of bitvectors.
(define (dot-product-spec V-1 V-2)
  (assert (= (length V-1) (length V-2)))
    (make-bv-list-zeros 4)
  ;(for* ([i V-1]
   ;      [j V-2])
          (vector-reduce-sum (vector-multiply V-1 V-2)))



(define (print v)
  (pretty-print v) v)

;Generate program skecth for dot product
(define (dot-product-shuffle-sketch V-1 V-2 iterations)
  ;Porgram preamble to define the inputs
  (define preamble
    (list
     (vec-extern-decl 'A (length V-1) input-tag)
     (vec-extern-decl 'B (length V-2) input-tag)
     (vec-extern-decl 'C 4 output-tag)
     (vec-decl 'reg-C (current-reg-size))))
  
  (define-values (C-reg-ids C-reg-loads C-reg-stores)
    (partition-bv-list 'C 4))
  
  ; Compute description for the sketch
  (define (compute-gen iteration shufs)
    ; Assumes that shuffle-gen generated three shuffle vectors
    (match-define (list shuf-A shuf-B) shufs)

    ; Shuffle the inputs with symbolic shuffle vectors
    (define input-shuffle
      (list
       (vec-shuffle 'reg-A shuf-A (list 'A))
       (vec-shuffle 'reg-B shuf-B (list 'B))))
    
    ; Use choose* to select an output register to both read and write
    (define output-dot-product
      (apply choose*
             (map (lambda (out-reg)
                    (list
                     (vec-write 'reg-C out-reg)
                     (vec-app 'mul-result 'vec-mul (list 'reg-A 'reg-B))
                     (vec-app 'out 'vec-sum (list 'mul-result))
                     (vec-write out-reg 'out)))
                  C-reg-ids)))
    
    (append input-shuffle output-dot-product))

  ; Shuffle vectors for each iteration
  (define shuffle-gen
    (symbolic-shuffle-gen 2))

  (prog
   (append preamble
           C-reg-loads
           (sketch-compute-shuffle-interleave
            shuffle-gen
            compute-gen
            iterations)
            C-reg-stores)))

; Run dot-prodcut sketch with symbolic inputs. If input bitvectors
; are missing, interp will generate freash symbolic values.
(define (run-dot-product-sketch sketch
                                    C-size
                                    cost-fn
                                    [V1 #f]
                                    [V2 #f])
  (define env (make-hash))
  (when (and V1 V2)
    (hash-set! env 'A V1)
    (hash-set! env 'B V2)
    (hash-set! env 'C (make-bv-list-zeros C-size)))

  (define-values (_ cost)
    (interp sketch
            env
            #:symbolic? #t
            #:cost-fn cost-fn
            #:fn-map (hash 'vec-mul vector-multiply
                           'vec-sum vector-reduce-sum)))

  (list (take (hash-ref env 'C) C-size) cost))

; Get statistics on a proposed synthesis solution

(define (get-statistics C-size sol reg-of)
  (let*
     ([regs-cost (last (run-dot-product-sketch
                      sol
                      C-size
                      (make-register-cost reg-of)))]
     [class-uniq-cost (last (run-dot-product-sketch
                            sol
                            C-size
                            (make-shuffle-unique-cost prefix-equiv)))])
    (pretty-print `(class-based-unique-idxs-cost: ,(bitvector->integer class-uniq-cost)))
    (pretty-print `(registers-touched-cost: ,(bitvector->integer regs-cost)))
    (pretty-print '-------------------------------------------------------)))
               
; Describe the configuration parameters for this benchmark
(define dot-product:keys
  (list 'V1 'V2 'iterations 'reg-size))
       
     
; Run dot product with the given spec.
; Requires that spec be a hash with all the keys describes in dot product:keys.
;(define (dot-product:run-experiment spec file-writer)
 ; (pretty-print (~a "Running dot prodcut with config: " spec))
  (define V1 2)
  (define V2 2)
  (define iterations 4)
  (define reg-size 4)
  (define pre-reg-of #t)

  (assert (equal? V1 V2)
          "dot product:run-experiment: Invalid bitvector sizes. V-1 not equal to V-2")

  ; Run the synthesis query
  (parameterize [(current-reg-size reg-size)]
    (define A (make-symbolic-bv-list (bitvector (value-fin)) 1))
    (define B (make-symbolic-bv-list (bitvector (value-fin)) 1))
    (define C-size 4)

    ; Generate sketch prog
    (define dot-product-v (dot-product-shuffle-sketch A B iterations))

    ; Determine whether to use the pre-computed register-of uninterpreted
    ; function, or pass the implementation to the solver directly
    ; assume is a list of booleans to be asserted, reg-of specifies which function
    ; to use for that computation
    (define-values (assume reg-of)
      (if pre-reg-of
          (build-register-of-map)
          (values (list) reg-of-idx)))

    ; Define the cost function
    (define (cost-fn)
      (let ([cost-1 (make-shuffle-unique-cost prefix-equiv)]
            [cost-2 (make-register-cost reg-of)])
        (lambda (inst env)
          (bvadd (cost-1 inst env) (cost-2 inst env)))))

    ;Create function for sketch evaluation
    (define (sketch-func args)
      (apply (curry run-dot-product-sketch
                    dot-product-v
                    C-size
                    (cost-fn))
             args))

     ; Functionalize spec for minimization prog
    (define (spec-func args)
      (apply dot-product-spec args))

    ; Get a generator back from the synthesis procedure
    (define model-generator
      (synth-prog spec-func
                  sketch-func
                  (list A B)
                  #:get-inps (lambda (args) (flatten (map unbox args)))
                  #:min-cost (bv-cost 0)
                  #:assume assume))

    ; Keep minimizing solution in the synthesis procedure and generating new
    ; solutions.
    (for ([(model cost) (sol-producer model-generator)])
      (if (sat? model)
        (let ([prog (evaluate dot-product-v model)])
          ;(file-writer prog cost)
          (pretty-print (concretize-prog prog)))
        (pretty-print (~a "failed to find solution: " model)))))