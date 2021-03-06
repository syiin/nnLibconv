import { NamedTensorMap, NamedTensorsMap, TensorInfo } from '../data/types';
import { Graph } from '../operations/types';
export declare class GraphExecutor {
    private graph;
    private compiledOrder;
    private _weightMap;
    private weightIds;
    private placeholders;
    private _outputs;
    weightMap: NamedTensorsMap;
    readonly inputs: TensorInfo[];
    readonly outputs: TensorInfo[];
    readonly inputNodes: string[];
    readonly outputNodes: string[];
    constructor(graph: Graph);
    readonly isControlFlowModel: boolean;
    private compile();
    execute(inputs: NamedTensorsMap, outputs?: string | string[]): NamedTensorMap;
    executeAsync(inputs: NamedTensorsMap, outputs?: string | string[]): Promise<NamedTensorMap>;
    private executeWithControlFlow(inputs, context);
    private findOutputs(tensorMap, context, outputs?);
    dispose(): void;
    private checkInputShapeAndType(inputs);
    private checkInput(inputs);
}
