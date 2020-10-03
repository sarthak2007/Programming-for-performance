import java.util.*;
import static java.util.stream.Collectors.*;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import org.antlr.v4.runtime.tree.ParseTreeProperty;
import org.antlr.v4.runtime.tree.TerminalNode;

// FIXME: You should limit your implementation to this class. You are free to add new auxilliary classes. You do not need to touch the LoopNext.g4 file.
class LoopData{
    String variableName;
    long bound, stride;
    LoopData(String variableName, long bound, long stride){
        this.variableName = variableName;
        this.bound = bound;
        this.stride = stride;
    }
}

class Analysis extends LoopNestBaseListener {

    // Possible types
    enum Types {
        Byte, Short, Int, Long, Char, Float, Double, Boolean, String
    }

    // Type of variable declaration
    enum VariableType {
        Primitive, Array, Literal
    }

    // Types of caches supported
    enum CacheTypes {
        DirectMapped, SetAssociative, FullyAssociative,
    }

    // auxilliary data-structure for converting strings
    // to types, ignoring strings because string is not a
    // valid type for loop bounds
    final Map<String, Types> stringToType = Collections.unmodifiableMap(new HashMap<String, Types>() {
        private static final long serialVersionUID = 1L;

        {
            put("byte", Types.Byte);
            put("short", Types.Short);
            put("int", Types.Int);
            put("long", Types.Long);
            put("char", Types.Char);
            put("float", Types.Float);
            put("double", Types.Double);
            put("boolean", Types.Boolean);
        }
    });

    // auxilliary data-structure for mapping types to their byte-size
    // size x means the actual size is 2^x bytes, again ignoring strings
    final Map<Types, Integer> typeToSize = Collections.unmodifiableMap(new HashMap<Types, Integer>() {
        private static final long serialVersionUID = 1L;

        {
            put(Types.Byte, 0);
            put(Types.Short, 1);
            put(Types.Int, 2);
            put(Types.Long, 3);
            put(Types.Char, 1);
            put(Types.Float, 2);
            put(Types.Double, 3);
            put(Types.Boolean, 0);
        }
    });

    // Map of cache type string to value of CacheTypes
    final Map<String, CacheTypes> stringToCacheType = Collections.unmodifiableMap(new HashMap<String, CacheTypes>() {
        private static final long serialVersionUID = 1L;

        {
            put("FullyAssociative", CacheTypes.FullyAssociative);
            put("SetAssociative", CacheTypes.SetAssociative);
            put("DirectMapped", CacheTypes.DirectMapped);
        }
    });

    /* started editing */
    HashMap<String, Long> identifiers = new HashMap<String, Long>();
    String cacheType;

    HashMap<String, String> arrayTypes = new HashMap<String, String>();
    HashMap<String, ArrayList<Long>> arrayDims = new HashMap<String, ArrayList<Long>>();
    HashMap<String, ArrayList<String>> arrayAccessDims = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<LoopData>> arrayAccessLoops = new HashMap<String, ArrayList<LoopData>>();

    ArrayList<Long> dims = new ArrayList<Long>();
    ArrayList<LoopData> loopDataList = new ArrayList<LoopData>();

    List<HashMap<String, Long>> result = new ArrayList<HashMap<String, Long>>();

    /* ended editing */

    public Analysis() {
    }

    // FIXME: Feel free to override additional methods from
    // LoopNestBaseListener.java based on your needs.
    // Method entry callback
    @Override
    public void enterMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
        // System.out.println("enterMethodDeclaration");
        identifiers.clear();
        arrayTypes.clear();
        arrayDims.clear();
        arrayAccessDims.clear();
        arrayAccessLoops.clear();
        dims.clear();
        loopDataList.clear();
    }

    static long power(long num){
        long ret = 1;
        while(num != 0){
            ret *= 2;
            num--;
        }
        return ret;
    }

    static long ceil(long num, long den){
        if(num < den)
            return 1;
        return num/den;
    }

    static long max(long a, long b){
        return a > b ? a : b;
    }

    // End of testcase
    @Override
    public void exitMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
        // display values starts

        // for(String key:identifiers.keySet()){
        //     System.out.println(key + " = " + identifiers.get(key));
        // }
        // System.out.println("cacheType = " + cacheType + "\n");
        // for(String key:arrayTypes.keySet()){
        //     System.out.print(arrayTypes.get(key) + " " + key);
        //     for(Long var : arrayDims.get(key)){
        //         System.out.print("["+var+"]");
        //     }
        //     System.out.print("\n");
        // }
        // System.out.print("\n");
        // for(String key:arrayAccessDims.keySet()){
        //     System.out.print(key);
        //     for(String var : arrayAccessDims.get(key)){
        //         System.out.print("["+var+"]");
        //     }
        //     System.out.print("\n");
        //     for(LoopData var : arrayAccessLoops.get(key)){
        //         System.out.println(var.variableName + " " + var.bound + " " + var.stride);
        //     }
        //     System.out.print("\n");
        // }

        // display values ends

        HashMap<String, Long> cacheMisses = new HashMap<String, Long>();
        for(String arrayName : arrayAccessDims.keySet()){
            long misses = 1;
            long dataTypeSize = power(typeToSize.get(stringToType.get(arrayTypes.get(arrayName))));
            long cacheSize = power(identifiers.get("cachePower"));
            long blockSize = power(identifiers.get("blockPower"));
            long setSize = 1;
            if(cacheType.equals("\"FullyAssociative\"")){
                setSize = cacheSize/blockSize;
            }
            else if(cacheType.equals("\"SetAssociative\"")){
                setSize = identifiers.get("setSize");
            }

            if(dataTypeSize > blockSize){
                blockSize = dataTypeSize;
                if(dataTypeSize > cacheSize/setSize){
                    setSize = cacheSize/dataTypeSize;
                }
            }
            long words = blockSize/dataTypeSize;
            long rows = cacheSize/(setSize * blockSize);

            int dimension = arrayAccessDims.get(arrayName).size();
            long[] prod = new long[dimension];
            long[] bound = new long[dimension];
            long[] stride = new long[dimension];
            String[] loopVarNames = new String[dimension];
            for(int i=0;i<dimension;i+=1){
                prod[i] = 1;
                bound[i] = 0;
                stride[i] = 0;
            }
            int flag = 0;
            for(LoopData var : arrayAccessLoops.get(arrayName)){
                if(arrayAccessDims.get(arrayName).contains(var.variableName)){
                    loopVarNames[flag] = var.variableName;
                    bound[flag] = var.bound;
                    stride[flag] = var.stride;
                    if(flag == dimension - 1)
                        break;
                    flag++;
                }
                else
                    prod[flag] *= ceil(var.bound, var.stride);
            }

            long[] dim = new long[dimension];
            long arraySize = dataTypeSize;
            for(int i=0; i<dimension; i++){
                dim[i] = arrayDims.get(arrayName).get(i);
                arraySize *= dim[i];
            }
            long[] M = new long[dimension];

            if(dimension == 1){
                M[0] = max(words, stride[0]);
                misses = ceil(bound[0], M[0]);
                if(arraySize > cacheSize){
                    if(ceil(bound[0], max(M[0], rows * words)) > setSize)
                        misses *= prod[0];
                }                
            }
            else if(dimension == 2){
                // i j variant
                if(loopVarNames[0].equals(arrayAccessDims.get(arrayName).get(0))){
                    M[1] = max(words, stride[1]);
                    M[0] = max(words/dim[1], stride[0]);

                    misses = ceil(bound[1], M[1]) * ceil(bound[0], M[0]);
                    if(arraySize > cacheSize){
                        if(ceil(bound[1], max(M[1], rows * words)) <= setSize){
                            if((M[0]*dim[1] < rows*words && ceil(bound[1], max(words*rows, M[1]))*ceil(bound[0], (rows*words)/dim[1]) 
                            <= setSize) || 
                            (M[0]*dim[1] >= rows*words && ceil(bound[1], max(words*rows, M[1]))*
                            ceil(bound[0], M[0]) <= setSize))
                                misses *= 1;
                            else
                                misses *= prod[0];
                        }
                        else{
                            misses *= (prod[0] * prod[1]);
                        }
                    }
                }

                // j i variant
                else{
                    M[0] = max(words, stride[0]);
                    M[1] = max(words/dim[1], stride[1]);

                    misses = ceil(bound[0], M[0]) * ceil(bound[1], M[1]);
                    if(arraySize > cacheSize){
                        if((M[1]*dim[1] < rows*words && ceil(bound[1], (rows*words)/dim[1]) 
                        <= setSize) || 
                        (M[1]*dim[1] >= rows*words && ceil(bound[1], M[1]) <= setSize)){
                            if((M[1]*dim[1] < rows*words && ceil(bound[0], max(words*rows, M[0]))*ceil(bound[1], (rows*words)/dim[1]) 
                            <= setSize) || 
                            (M[1]*dim[1] >= rows*words && ceil(bound[0], max(words*rows, M[0]))*
                            ceil(bound[1], M[1]) <= setSize))
                                misses *= 1;
                            else
                                misses *= prod[0];
                        }
                        else{
                            misses = ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];
                        }
                    }
                }

            }
            else if(dimension == 3){
                // i j k and j i k variant 
                if(loopVarNames[2].equals(arrayAccessDims.get(arrayName).get(2))){
                    M[2] = max(words, stride[2]);

                    // i j k variant
                    if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(1))){
                        M[1] = max(words/dim[2], stride[1]);
                        M[0] = max(words/(dim[2]*dim[1]), stride[0]);
                    }
                    // j i k variant
                    else{
                        M[1] = max(words/(dim[2]*dim[1]), stride[1]);
                        M[0] = max(words/dim[2], stride[0]);
                    }

                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], M[0]);
                    if(arraySize > cacheSize){
                        if(ceil(bound[2], max(M[2], rows * words)) <= setSize){

                            // i j k variant
                            if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(1))){

                                if((M[1]*dim[2] < rows*words && ceil(bound[2], max(words*rows, M[2]))*ceil(bound[1], (rows*words)/dim[2]) 
                                <= setSize) || 
                                (M[1]*dim[2] >= rows*words && ceil(bound[2], max(words*rows, M[2]))*
                                ceil(bound[1], M[1]) <= setSize)){

                                    long inner = 0;
                                    if(M[1]*dim[2] < rows*words)
                                        inner = ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], (rows*words)/dim[2]);
                                    else
                                        inner = ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], M[1]);
                                        
                                    if((M[0]*dim[1]*dim[2] < rows*words && inner * ceil(bound[0], (rows*words)/(dim[1]*dim[2])) 
                                    <= setSize) || 
                                    (M[0]*dim[1]*dim[2] >= rows*words && inner * ceil(bound[0], M[0]) <= setSize)){

                                        misses *= 1;
                                    }
                                    else
                                        misses *= prod[0];
                                }
                                else
                                    misses *= (prod[0] * prod[1]);
                            }

                            // j i k variant
                            else{
                                if(M[1]*dim[2]*dim[1] < rows*words && ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], (rows*words)/(dim[1]*dim[2])) <= setSize){
                                    long temp = ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], (rows*words)/(dim[1]*dim[2])) * ceil(bound[0], (rows*words)/dim[2]);
                                    if(temp > setSize)
                                        misses *= prod[0];
                                }
                                else if(M[1]*dim[2]*dim[1] >= rows*words && ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], M[1]) <= setSize){
                                    long temp = ceil(bound[2], max(rows*words, M[2])) * ceil(bound[1], M[1]);
                                    if(M[0]*dim[2] < rows*words)
                                        temp *= ceil(bound[0], (rows*words)/dim[2]);
                                    else
                                        temp *= ceil(bound[0], M[0]);
                                    if(temp > setSize)
                                        misses *= prod[0];    
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];
                                }
                            }
                        }
                        else
                            misses *= (prod[0] * prod[1] * prod[2]);
                    }
                }

                // i k j and k i j variant
                else if(loopVarNames[2].equals(arrayAccessDims.get(arrayName).get(1))){
                    M[2] = max(words/dim[2], stride[2]);

                    // i k j variant
                    if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(2))){
                        M[1] = max(words, stride[1]);
                        M[0] = max(words/(dim[2]*dim[1]), stride[0]);
                    }
                    // k i j variant
                    else{
                        M[1] = max(words/(dim[2]*dim[1]), stride[1]);
                        M[0] = max(words, stride[0]);
                    }

                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], M[0]);
                    if(arraySize > cacheSize){
                        
                        // i k j variant
                        if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(2))){

                            if(M[2]*dim[2] < rows*words && ceil(bound[2], (rows*words)/dim[2]) <= setSize){
                                long inner = ceil(bound[2], (rows*words)/dim[2]) * ceil(bound[1], max(rows*words, M[1]));
                                if(inner <= setSize){ 
                                    if(M[0]*dim[1]*dim[2] < rows*words && inner * ceil(bound[0], (rows*words)/(dim[1]*dim[2])) <= setSize){
                                        misses *= 1;
                                    }
                                    else if(M[0]*dim[1]*dim[2] >= rows*words && inner * ceil(bound[0], M[0]) <= setSize){
                                        misses *= 1;
                                    }
                                    else{
                                        misses *= prod[0];
                                    }
                                }
                                else{
                                    misses *= (prod[0] * prod[1]);
                                }
                            }
                            else if(M[2]*dim[2] >= rows*words && ceil(bound[2], M[2]) <= setSize){
                                long inner = ceil(bound[2], M[2]) * ceil(bound[1], max(rows*words, M[1]));
                                if(inner <= setSize){
                                    if(inner*ceil(bound[0], M[0]) > setSize){
                                        misses *= prod[0];
                                    }
                                }
                                else{
                                    misses *= (prod[0] * prod[1]);
                                }
                            }
                            else{
                                misses = ceil(bound[2], M[2]) * ceil(bound[1], stride[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1] * prod[2];
                            }

                        }

                        // k i j variant
                        else{
                            if(M[2]*dim[2] < rows*words && ceil(bound[2], (rows*words)/dim[2]) <= setSize){
                                long inner = ceil(bound[2], (rows*words)/dim[2]);
                                if(M[1]*dim[1]*dim[2] < rows*words && inner * ceil(bound[1], (rows*words)/(dim[1]*dim[2])) <= setSize){
                                    inner *= ceil(bound[1], (rows*words)/(dim[1]*dim[2]));
                                    if(inner * ceil(bound[0], max(rows*words, M[0])) > setSize)
                                        misses *= prod[0];
                                }
                                else if(M[1]*dim[1]*dim[2] >= rows*words && inner * ceil(bound[1], M[1]) <= setSize){
                                    inner *= ceil(bound[1], M[1]);
                                    if(inner * ceil(bound[0], max(rows*words, M[0])) > setSize)
                                        misses *= prod[0];
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];
                                }
                            }
                            else if(M[2]*dim[2] >= rows*words && ceil(bound[2], M[2]) <= setSize){
                                long inner = ceil(bound[2], M[2]) * ceil(bound[1], M[1]);
                                if(inner <= setSize){
                                    if(inner * ceil(bound[0], max(rows*words, M[0])) > setSize)
                                        misses *= prod[0];
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];
                                }
                            }
                            else{
                                misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1] * prod[2];
                            }
                        }
                        
                    }
                }

                // j k i and k j i variant
                else if(loopVarNames[2].equals(arrayAccessDims.get(arrayName).get(0))){
                    M[2] = max(words/(dim[2]*dim[1]), stride[2]);

                    // j k i variant
                    if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(2))){
                        M[1] = max(words, stride[1]);
                        M[0] = max(words/dim[2], stride[0]);
                    }
                    // k j i variant
                    else{
                        M[1] = max(words/dim[2], stride[1]);
                        M[0] = max(words, stride[0]);
                    }

                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], M[0]);
                    if(arraySize > cacheSize){
                        
                        // j k i variant
                        if(loopVarNames[1].equals(arrayAccessDims.get(arrayName).get(2))){

                            if(M[2]*dim[1]*dim[2] < rows*words && ceil(bound[2], (rows*words)/(dim[2]*dim[1])) <= setSize){
                                long inner = ceil(bound[2], (rows*words)/(dim[2]*dim[1])) * ceil(bound[1], max(rows*words, M[1]));

                                if(inner <= setSize){
                                    if(inner * ceil(bound[0], (rows*words)/dim[2]) > setSize)
                                        misses *= prod[0];
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1]; 
                                }
                                
                            }
                            else if(M[2]*dim[1]*dim[2] >= rows*words && ceil(bound[2], M[2]) <= setSize){
                                long inner = ceil(bound[2], M[2]) * ceil(bound[1], max(rows*words, M[1]));
                                if(inner <= setSize){
                                    if(M[0]*dim[2] < rows*words && inner * ceil(bound[0], (rows*words)/dim[2]) <= setSize){
                                        misses *= 1;
                                    }
                                    else if(M[0]*dim[2] >= rows*words && inner * ceil(bound[0], M[0]) <= setSize){
                                        misses *= 1;
                                    }
                                    else{
                                        misses *= prod[0];
                                    }
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];    
                                }
                            }
                            else{
                                misses = ceil(bound[2], M[2]) * ceil(bound[1], stride[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1] * prod[2];
                            }

                        }

                        // k j i variant
                        else{
                            
                            if(M[2]*dim[1]*dim[2] < rows*words && ceil(bound[2], (rows*words)/(dim[2]*dim[1])) <= setSize){
                                long inner = ceil(bound[2], (rows*words)/(dim[2]*dim[1])) * ceil(bound[1], (rows*words)/dim[2]);

                                if(inner <= setSize){
                                    if(inner * ceil(bound[0], max(words*rows, M[0])) > setSize)
                                        misses *= prod[0];
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1]; 
                                }
                            }
                            else if(M[2]*dim[1]*dim[2] >= rows*words && ceil(bound[2], M[2]) <= setSize){
                                long inner = ceil(bound[2], M[2]);
                                if(M[1]*dim[2] < rows*words && inner * ceil(bound[1], (rows*words)/dim[2]) <= setSize){
                                    inner *= ceil(bound[1], (rows*words)/dim[2]);
                                    if(inner * ceil(bound[0], max(words*rows, M[0])) > setSize){
                                        misses *= prod[0];
                                    }
                                }
                                else if(M[1]*dim[2] >= rows*words && inner * ceil(bound[1], M[1]) <= setSize){
                                    inner *= ceil(bound[1], M[1]);
                                    if(inner * ceil(bound[0], max(words*rows, M[0])) > setSize){
                                        misses *= prod[0];
                                    }
                                }
                                else{
                                    misses = ceil(bound[2], M[2]) * ceil(bound[1], M[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1];
                                }
                            }
                            else{
                                misses = ceil(bound[2], M[2]) * ceil(bound[1], stride[1]) * ceil(bound[0], stride[0]) * prod[0] * prod[1] * prod[2];
                            }
                        }
                        
                    }
                }
            }

            cacheMisses.put(arrayName, misses);
        }

        result.add(cacheMisses);
        // System.out.println("exitMethodDeclaration");
    }

    @Override
    public void exitTests(LoopNestParser.TestsContext ctx) {
        // int cnt=1;
        // for(HashMap<String, Long> var : result){
        //     System.out.println("Testcase " + cnt++ + ":");
        //     for(String name : var.keySet()){
        //         System.out.println(name + " -> " + var.get(name));
        //     }
        //     System.out.print("\n");
        // }
        try {
            FileOutputStream fos = new FileOutputStream("Results.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            // FIXME: Serialize your data to a file
            oos.writeObject(result);
            oos.close();
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public void exitLocalVariableDeclaration(LoopNestParser.LocalVariableDeclarationContext ctx) {
        String variableName = ctx.getChild(1).getChild(0).getText();
        String variableType = ctx.getChild(0).getText();

        if(ctx.unannType()!=null && ctx.unannType().unannArrayType()!=null){
            variableType = ctx.getChild(0).getChild(0).getChild(0).getText();
            arrayTypes.put(variableName, variableType);
            ArrayList<Long> cloneDims = new ArrayList<Long>();
            for(Long var : dims)
                cloneDims.add(var);
            arrayDims.put(variableName, cloneDims);
            dims.clear();
        }
        else{
            String variableValue = ctx.getChild(1).getChild(2).getText(); 
            if(variableName.equals("cacheType")){
                cacheType = variableValue;
            }
            else{
                if(variableType.equals("long") || variableType.equals("int") || variableType.equals("short") || variableType.equals("byte")){
                    identifiers.put(variableName, Long.parseLong(variableValue));
                }
            }
        }
    }

    @Override
    public void exitDimExpr(LoopNestParser.DimExprContext ctx) {
        String variableValue = ctx.getChild(1).getText();
        if(ctx.IntegerLiteral() == null){
            dims.add(identifiers.get(variableValue));
        }
        else{
            dims.add(Long.parseLong(variableValue));
        }
    }

    @Override
    public void enterForStatement(LoopNestParser.ForStatementContext ctx) {
        String variableName = ctx.forInit().localVariableDeclaration().variableDeclarator().
            variableDeclaratorId().getText();
        Long condition;
        String conditionValue = ctx.forCondition().relationalExpression().getChild(2).getText();
        if(ctx.forCondition().relationalExpression().IntegerLiteral() == null){
            condition = identifiers.get(conditionValue);
        }
        else{
            condition = Long.parseLong(conditionValue);
        }
        Long stride;
        String strideValue = ctx.forUpdate().simplifiedAssignment().getChild(2).getText();
        if(ctx.forUpdate().simplifiedAssignment().IntegerLiteral() == null){
            stride = identifiers.get(strideValue);
        }
        else{
            stride = Long.parseLong(strideValue);
        }
        LoopData loopData = new LoopData(variableName, condition, stride);
        loopDataList.add(loopData);
    }

    @Override
    public void exitForStatement(LoopNestParser.ForStatementContext ctx) {
        loopDataList.remove(loopDataList.size()-1);
    }

    @Override
    public void exitArrayAccess(LoopNestParser.ArrayAccessContext ctx) {
        int children = ctx.getChildCount();
        String arrayName = ctx.getChild(0).getText();
        ArrayList<LoopData> cloneLoopData = new ArrayList<LoopData>();
        for(LoopData var : loopDataList)
            cloneLoopData.add(var);
        arrayAccessLoops.put(arrayName, cloneLoopData);

        ArrayList<String> accessDims = new ArrayList<String>();
        accessDims.add(ctx.getChild(2).getText());
        if(children >= 7){
            accessDims.add(ctx.getChild(5).getText());
        }
        if(children == 10){
            accessDims.add(ctx.getChild(8).getText());
        }
        arrayAccessDims.put(arrayName, accessDims);
    }

    @Override
    public void exitArrayAccess_lfno_primary(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
        int children = ctx.getChildCount();
        String arrayName = ctx.getChild(0).getText();
        ArrayList<LoopData> cloneLoopData = new ArrayList<LoopData>();
        for(LoopData var : loopDataList)
            cloneLoopData.add(var);
        arrayAccessLoops.put(arrayName, cloneLoopData);

        ArrayList<String> accessDims = new ArrayList<String>();
        accessDims.add(ctx.getChild(2).getText());
        if(children >= 7){
            accessDims.add(ctx.getChild(5).getText());
        }
        if(children == 10){
            accessDims.add(ctx.getChild(8).getText());
        }
        arrayAccessDims.put(arrayName, accessDims);
    }

}
