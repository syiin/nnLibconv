import * as tfc from '@tensorflow/tfjs-core';
import { ExecutionContext } from '../../executor/execution_context';
import { executeOp } from './arithmetic_executor';
import { createTensorAttr } from './test_helper';
describe('arithmetic', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(1)];
    var context = new ExecutionContext({});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'arithmetic',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: { a: createTensorAttr(0), b: createTensorAttr(1) },
            children: []
        };
    });
    describe('executeOp', function () {
        ['add', 'mul', 'div', 'sub', 'maximum', 'minimum', 'pow',
            'squaredDifference', 'mod', 'floorDiv']
            .forEach((function (op) {
            it('should call tfc.' + op, function () {
                var spy = spyOn(tfc, op);
                node.op = op;
                executeOp(node, { input1: input1, input2: input2 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], input2[0]);
            });
        }));
    });
});
//# sourceMappingURL=arithmetic_executor_test.js.map