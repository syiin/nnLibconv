import * as tfc from '@tensorflow/tfjs-core';
import { test_util } from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './graph_executor';
import { createNumberAttr, createStrAttr, createTensorAttr, createTensorsAttr } from './test_helper';
describe('graph', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var input2 = [tfc.tensor1d([1])];
    var input3 = [tfc.tensor3d([1, 1, 1, 2, 2, 2], [1, 2, 3])];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'graph',
            inputNames: [],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('const', function () {
            it('should return input', function () {
                node.op = 'const';
                expect(executeOp(node, { input1: input1 }, context)).toEqual(input1);
            });
        });
        describe('placeholder', function () {
            it('should return input', function () {
                node.op = 'placeholder';
                expect(executeOp(node, { input1: input1 }, context)).toEqual(input1);
            });
            it('should return default if input not set', function () {
                node.inputNames = ['input2'];
                node.op = 'placeholder';
                node.params.default = createTensorAttr(0);
                expect(executeOp(node, { input2: input2 }, context)).toEqual(input2);
            });
        });
        describe('identity', function () {
            it('should return input', function () {
                node.inputNames = ['input'];
                node.params.x = createTensorAttr(0);
                node.op = 'identity';
                expect(executeOp(node, { input: input1 }, context)).toEqual(input1);
            });
        });
        describe('snapshot', function () {
            it('should return input', function () {
                node.inputNames = ['input'];
                node.params.x = createTensorAttr(0);
                node.op = 'snapshot';
                var result = executeOp(node, { input: input1 }, context)[0];
                expect(result.rank).toEqual(input1[0].rank);
                test_util.expectArraysClose(result, [1]);
            });
        });
        describe('shape', function () {
            it('should return shape', function () {
                node.inputNames = ['input'];
                node.params.x = createTensorAttr(0);
                node.op = 'shape';
                expect(Array.prototype.slice.call(executeOp(node, { input: input3 }, context)[0]
                    .dataSync()))
                    .toEqual([1, 2, 3]);
            });
        });
        describe('size', function () {
            it('should return size', function () {
                node.inputNames = ['input'];
                node.params.x = createTensorAttr(0);
                node.op = 'size';
                expect(Array.prototype.slice.call(executeOp(node, { input: input3 }, context)[0]
                    .dataSync()))
                    .toEqual([6]);
            });
        });
        describe('rank', function () {
            it('should return rank', function () {
                node.inputNames = ['input'];
                node.params.x = createTensorAttr(0);
                node.op = 'rank';
                expect(Array.prototype.slice.call(executeOp(node, { input: input3 }, context)[0]
                    .dataSync()))
                    .toEqual([3]);
            });
        });
        describe('noop', function () {
            it('should return empty', function () {
                node.op = 'noop';
                expect(executeOp(node, {}, context)).toEqual([]);
            });
        });
    });
    describe('print', function () {
        it('should return empty', function () {
            node.op = 'print';
            node.inputNames = ['input1', 'input2'];
            node.params.x = createTensorAttr(0);
            node.params.data = createTensorsAttr(1, 1);
            node.params.message = createStrAttr('message');
            node.params.summarize = createNumberAttr(1);
            spyOn(console, 'log');
            spyOn(console, 'warn');
            expect(executeOp(node, { input1: input1, input2: input2 }, context)).toEqual(input1);
            expect(console.warn).toHaveBeenCalled();
            expect(console.log).toHaveBeenCalledWith('message');
            expect(console.log).toHaveBeenCalledWith([1]);
        });
    });
    describe('stopGradient', function () {
        it('should return input', function () {
            node.inputNames = ['input'];
            node.params.x = createTensorAttr(0);
            node.op = 'stopGradient';
            expect(executeOp(node, { input: input1 }, context)).toEqual(input1);
        });
    });
    describe('fakeQuantWithMinMaxVars', function () {
        it('should return input', function () {
            node.inputNames = ['input'];
            node.params.x = createTensorAttr(0);
            node.op = 'fakeQuantWithMinMaxVars';
            expect(executeOp(node, { input: input1 }, context)).toEqual(input1);
        });
    });
});
//# sourceMappingURL=graph_executor_test.js.map