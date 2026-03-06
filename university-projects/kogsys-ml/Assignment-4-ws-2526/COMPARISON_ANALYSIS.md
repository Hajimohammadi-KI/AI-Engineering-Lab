# Assignment 4 - Solution Comparison
## Group 2 vs. student_version_en

---

## **SECTION 1: CONVOLUTIONS** ✅ IDENTICAL

### 04.1.1 - Full Convolution (stride=1, padding=1)
- **Status**: ✅ **IDENTICAL**
- Both solutions produce the same final result matrix:
```
[[10.0,   9.5,   5.5,  13.0],
 [3.5,   11.0,  18.0,  21.0],
 [-1.0,  12.5,  13.5,  15.5],
 [3.0,    9.5,   7.5,  -0.5]]
```

### 04.1.2 - Downsampling Convolution (stride=2)
- **Status**: ✅ **IDENTICAL**
- Both produce the same 2×2 output:
```
[[10.0,  5.5],
 [-1.0, 13.5]]
```

---

## **SECTION 2: REINFORCEMENT LEARNING** ⚠️ SIMILAR BUT DIFFERENT QUALITY

### 04.2.1 - Domain Class Implementation

| Aspect | Group 2 | student_version_en |
|--------|---------|-------------------|
| **States** | 10 states (simpler structure) | More granular state space (PACKAGE_DONE added) |
| **Transitions** | Basic dictionary | Detailed graph representation |
| **Documentation** | Minimal | Comprehensive docstrings |
| **Helper Methods** | Basic methods | More utility methods (get_reward, is_terminal, get_random_start_state) |
| **Code Quality** | Functional | Production-quality |
| **Reward Values** | -10 for pickup | -3 for pickup |

**Key Difference**: 
- Group 2 uses `-10` for pickup reward
- student_version_en uses `-3` for pickup reward
- Both approaches are valid, but different penalty magnitudes will affect learning

### 04.2.2 - Constructor
| Aspect | Group 2 | student_version_en |
|--------|---------|-------------------|
| **Parameters** | domain only | domain + gamma parameter |
| **Gamma Default** | N/A (hardcoded 0.9 in training) | 0.9 (configurable) |
| **Docstring** | Present | Detailed with parameter descriptions |

### 04.2.3-4 - Helper & Training Methods

**Both implementations are FUNCTIONALLY CORRECT**

| Feature | Group 2 | student_version_en |
|---------|---------|-------------------|
| **init_df()** | ✅ Correct | ✅ Correct |
| **apply_random_action()** | Simple, works | More error handling |
| **get_reward()** | Functional | Documented |
| **update()** | ✅ Correct | ✅ Correct + docstring |
| **finalize()** | ✅ Correct | ✅ Correct + docstring |
| **is_goal_state()** | ✅ Correct | ✅ Correct + docstring |
| **do_epoch()** | ✅ Correct algorithm | ✅ Correct + better comments |
| **train()** | Returns q_table | Returns q_table + progress output |
| **optimal()** | Returns path | Returns (path, total_reward) |

### 04.2.5 - Application Methods

**Group 2:**
- `optimal()`: Returns just the path
- `paths()`: Prints and returns all optimal paths as dictionary

**student_version_en:**
- `optimal()`: Returns `(path, total_reward)` - more informative
- `paths()`: Returns dictionary with paths and rewards for all states

**Verdict**: ⚠️ Both functionally work, but student_version_en provides richer information

---

## **SECTION 3: HIDDEN MARKOV MODELS** ✅ NEARLY IDENTICAL

### 04.3.1 - Transition Matrix
- **Status**: ✅ **IDENTICAL**
- Both provide the 5×5 transition matrix correctly

### 04.3.2 - Emission Probabilities
- **Status**: ✅ **IDENTICAL**
- Both provide the 5×4 emission matrix correctly
- Both use $\frac{0.2}{3} \approx 0.0667$ for incorrect observations

### 04.3.3 - Forward Algorithm Manual Calculation
- **Status**: ✅ **IDENTICAL**
- Both calculate:
  - $\alpha_1 = [0.8, 0, 0, 0, 0]$ ✅
  - $\alpha_2 = [0, 0.0268, 0.0268, 0, 0]$ ✅
  - $P(O|\lambda) = 0.0536$ ✅

### 04.3.4 - Viterbi Algorithm Implementation

**Both are functionally correct, but differ in presentation:**

| Feature | Group 2 | student_version_en |
|---------|---------|-------------------|
| **Constructor** | Takes 5 params | Takes 6 params (plus eta) |
| **Viterbi Method** | Core algorithm only | Algorithm + detailed debug output |
| **Documentation** | Brief comments | Extensive docstrings with step numbers |
| **Output** | Returns (path, prob) | Returns (path, prob) + prints detailed steps |
| **Test Case** | Simple test | Comprehensive test with interpretation |
| **Code Clarity** | Concise | Very detailed, step-by-step |

**Test Output Comparison:**
- Group 2: Minimal output, just final result
- student_version_en: Shows all initialization, recursion, termination, and backtracking steps

**Both produce correct result:**
```
State sequence: ['START', 'FLAT_PATH', 'PACKAGE', 'GOAL']
Probability: 0.2048
```

---

## **OVERALL ASSESSMENT**

### ✅ **Correctness: BOTH 84/84 POINTS**
- All mathematical calculations are correct
- All algorithms are properly implemented
- Both pass all test cases

### Code Quality Comparison:

| Aspect | Group 2 | student_version_en |
|--------|---------|-------------------|
| **Correctness** | ✅ 100% | ✅ 100% |
| **Documentation** | ⭐⭐⭐ Basic | ⭐⭐⭐⭐⭐ Excellent |
| **Code Clarity** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Error Handling** | ⭐⭐⭐ Basic | ⭐⭐⭐⭐ Good |
| **Debuggability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (detailed output) |
| **Reusability** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Best Practices** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## **KEY DIFFERENCES SUMMARY**

### 1. **Documentation**
- **Group 2**: Minimal comments
- **student_version_en**: Comprehensive docstrings following best practices

### 2. **Code Organization**
- **Group 2**: Compact, straightforward
- **student_version_en**: More modular, better separation of concerns

### 3. **Debugging Support**
- **Group 2**: Minimal output
- **student_version_en**: Extensive print statements for algorithm verification

### 4. **Domain Modeling**
- **Group 2**: Simpler state structure
- **student_version_en**: More detailed state representation

### 5. **Return Values**
- **Group 2**: Basic returns (paths only)
- **student_version_en**: Richer returns (paths + rewards)

---

## **RECOMMENDATIONS**

✅ **Both solutions are CORRECT and would receive full marks (84/84)**

**However:**
- **For production code**: Use student_version_en approach
- **For learning**: Group 2's simplicity is easier to understand
- **For teaching others**: student_version_en's documentation is superior
- **For grading**: Both deserve full credit but student_version_en shows stronger software engineering practices

---

## **VERDICT**

| Criterion | Result |
|-----------|--------|
| **Mathematical Correctness** | ✅ Both Perfect |
| **Algorithm Implementation** | ✅ Both Correct |
| **Code Quality** | ⭐ student_version_en slightly better |
| **Documentation** | ⭐ student_version_en significantly better |
| **Points Earned** | 🎯 Both: 84/84 |

