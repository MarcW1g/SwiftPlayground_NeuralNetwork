//: Playground - noun: a place where people can play

import UIKit

// MARK: - Array operations
extension Array where Iterator.Element == Double {
    static func *(left: [Double], right: [Double]) -> Double {
        var product: Double = 0.0

        for (i, key) in left.enumerated() {
            product += key * right[i]
        }

        return product
    }

    static func *(value: Double, right: [Double]) -> Double {
        var product: Double = 0.0

        for key in right {
            product += value * key
        }

        return product
    }

    static func *(left: [Double], right: Double) -> [Double] {
        var productArr: [Double] = []

        for key in left {
            productArr.append(right * key)
        }

        return productArr
    }

    static func -(left: [[Double]], right: [[Double]]) -> [[Double]] {
        var newArray: [[Double]] = []

        for (index, leftRow) in left.enumerated() {
            var row: [Double] = []
            for (item, leftItem) in leftRow.enumerated() {
                row.append(leftItem - right[index][item])
            }
            newArray.append(row)
        }

        return newArray
    }

    static func -(left: [Double], right: [Double]) -> [Double] {
        var newArray: [Double] = []

        for (index, leftItem) in left.enumerated() {
            newArray.append(leftItem - right[index])
        }

        return newArray
    }

    static func +(left: [Double], right: [Double]) -> [Double] {
        var newArray: [Double] = []

        for (index, leftItem) in left.enumerated() {
            newArray.append(leftItem + right[index])
        }

        return newArray
    }
}


class LinearAlgebra {
    func dotProduct(left:[[Double]], right:[[Double]]) -> [[Double]] {
        guard left[0].count ==  right.count else {
            print("Dimensions are incorrect")
            return [[]]
        }

        var dot: [[Double]] = []
        for (_, left_row) in left.enumerated() {
            var dotRow: [Double] = []
            for i in 0..<right[0].count {
                var newRow: [Double] = []
                for row in right {
                    newRow.append(row[i])
                }
                dotRow.append(left_row*newRow)
            }
            dot.append(dotRow)
        }

        return dot
    }

    func vectorTimes(left:[Double], right:[Double]) -> [Double] {
        var returnArray: [Double] = []
        for (i,l_item) in left.enumerated() {
            returnArray.append(l_item * right[i])
        }
        return returnArray
    }
}

class NeuralKit {
    // MARK: - Neural network

    // MARK: - Basic setup functions
    func sigmoid(_ X:[[Double]]) -> [[Double]] {
        var newArray: [[Double]] = []

        for row in X {
            var newRow: [Double] = []
            for col in row {
                newRow.append(1/(1+exp(-col)))
            }
            newArray.append(newRow)
        }
        return newArray
    }


    func addBias(_ X: [[Double]]) -> [[Double]] {
        var newArray: [[Double]] = []

        for row in X {
            var currentRow = row
            currentRow.insert(1.0, at: 0)
            newArray.append(currentRow)
        }

        return newArray
    }

    // This function creates a new matrix with doubles between -0.5 and 0.5
    func randomWeigts(nrOfInputNodes: Int, nrOfOutputNodes: Int) -> [[Double]] {
        var newArray: [[Double]] = []
        for _ in 0..<nrOfOutputNodes {
            var newRow: [Double] = []
            for _ in 0..<nrOfInputNodes+1 {
                newRow.append(drand48() - 0.5)
            }
            newArray.append(newRow)
        }
        return newArray
    }

    // MARK: - Functions to calculate the outputs for one layer

    func computeLayer(activations: [[Double]], weights: [[Double]]) -> [[Double]]{
        var newArray: [[Double]] = []

        for a_row in activations {
            var temp: [Double] = []
            for w_row in weights {
                temp.append(a_row * w_row)
            }
            newArray.append(temp)
        }
        return newArray
    }

    func oneLayerOut(X: [[Double]], weights: [[Double]]) -> [[Double]] {
        let biasedX = addBias(X)
        return self.sigmoid(computeLayer(activations: biasedX, weights: weights))
    }

    // MARK: - Functions to calculate the output for multiple layers
    func n_layerInit(size: [Int]) -> [[[Double]]] {
        var weights: [[[Double]]] = []

        for i in 0..<(size.count)-1 {
            weights.append(randomWeigts(nrOfInputNodes: size[i], nrOfOutputNodes: size[i+1]))
        }

        return weights
    }

    func n_layerOutput(X: [[Double]], weights: [[[Double]]]) -> [[Double]] {
        var retX = X

        for weight in weights {
            retX = oneLayerOut(X: retX, weights: weight)
        }
        return retX
    }

    // MARK: - Cost Function
    func costFunction(A: [[Double]], Y: [[Double]]) -> Double {
        var cost = 0.0
        for (i, y) in Y.enumerated() {
            for (k, a) in A[i].enumerated() {
                let yVal = y[k]
                cost += (yVal * log(a)) + ((1-yVal) * log(1-a))
            }
        }
        return -cost
    }

    // MARK: - Weight update function
    func updateWeights(A: [[Double]], delta: [[Double]], weights: [[Double]], rate: Double) -> [[Double]] {
        var newWeights: [[Double]] = []

        for i in 0..<delta[0].count {

            var gradient: [Double] = []
            for _ in 0..<A[0].count {
                gradient.append(0.0)
            }

            for (j, D) in delta.enumerated() {

                var newGradient: [Double] = []
                for item in A[j] {
                    newGradient.append(item * D[i])
                }

                for (index, item) in gradient.enumerated() {
                    gradient[index] = item + newGradient[index]
                }
            }

            for (index, item) in gradient.enumerated() {
                gradient[index] = item * rate
            }

            var currentWeight = weights[i]

            if currentWeight.count == gradient.count {
                for (index, item) in currentWeight.enumerated() {
                    currentWeight[index] = item - gradient[index]
                }
            } else {
                for (index, item) in currentWeight.enumerated() {
                    currentWeight[index] = item - gradient[0]
                }
            }

            newWeights.append(currentWeight)
        }
        return newWeights
    }

    // MARK: - Functions to calculate the delta's
    func outputDelta(A: [[Double]], Y: [[Double]]) -> [[Double]] {
        return A - Y
    }

    func hiddenLayer_delta(A: [[Double]], nextDelta: [[Double]], weights: [[Double]]) -> [[Double]] {

        var delta_hidden: [[Double]] = []

        for (i, A_row) in A.enumerated() {
            var new_A_row = A_row
            for (index, item) in A_row.enumerated() {
                new_A_row[index] = (1-item)*item
            }

            let partTwo = LinearAlgebra().dotProduct(left: [nextDelta[i]], right: weights)
            delta_hidden.append(LinearAlgebra().vectorTimes(left: new_A_row, right: partTwo[0]))
        }

        for (i, row) in delta_hidden.enumerated() {
            var newRow = row
            newRow.removeFirst()
            delta_hidden[i] = newRow
        }

        return delta_hidden // return the new alpha but without the bias
    }


    //MARK: - The functions for training
    func oneLayerTraining(data X: [[Double]], labels Y: [[Double]], weights: [[[Double]]], iterations iters: Int, learningRate rate: Double) -> [[Double]] {
        var new_weights = weights[0]

        var cost = 0.0

        for _ in 0..<iters {
            let networkOutput = self.n_layerOutput(X: X, weights: [new_weights])
            let oneLayerDelta = self.outputDelta(A: networkOutput, Y: Y)

            new_weights = self.updateWeights(A: self.addBias(X), delta: oneLayerDelta, weights: new_weights, rate: rate)

            cost = self.costFunction(A: networkOutput, Y: Y)
        }

        print(cost)
        return new_weights
    }

    func twoLayerTraining(data X: [[Double]], labels Y: [[Double]], weights0: [[[Double]]], weights1: [[[Double]]], iterations iters: Int, learningRate rate: Double) -> [[[Double]]] {

        var new_weights_0 = weights0[0]
        var new_weights_1 = weights1[0]

        var cost = 0.0

        for _ in 0..<iters {
            let hiddenLayerOut = self.oneLayerOut(X: X, weights: new_weights_0)
            let endLayerOutput = self.oneLayerOut(X: hiddenLayerOut, weights: new_weights_1)

            let nextDelta = self.outputDelta(A: endLayerOutput, Y: Y)
            let hiddenDelta = self.hiddenLayer_delta(A: self.addBias(hiddenLayerOut), nextDelta: nextDelta, weights: new_weights_1)

            new_weights_0 = self.updateWeights(A: self.addBias(X), delta: hiddenDelta, weights: new_weights_0, rate: rate)
            new_weights_1 = self.updateWeights(A: self.addBias(hiddenLayerOut), delta: nextDelta, weights: new_weights_1, rate: rate)

            cost = self.costFunction(A: endLayerOutput, Y: Y)
        }
        return [new_weights_0, new_weights_1]
    }
}

var network = NeuralKit()
network.oneLayerOut(X: [[1.0,2.0],[4.0,5.0],[7.0,8.0]], weights: [[0.1,0.1,0.2],[0.2,0.2,0.3]])
//network.times(value: 2.0, right: [2.0,3.0])



