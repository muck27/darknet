#include <jansson.h>

int test() {
    FILE* file = fopen("/clyde/content/trial/darknet_recv/data/json.txt", "r");
    char *line[256];
    char *input_input[256];
    int count=0;
    while (fgets(line, sizeof(line), file)) {
        strcpy(input_input, line);
        strtok(input_input, "\n");
        json_error_t error;
        json_t *root, *obj;
        int i;
        root = json_load_file(input_input, 0, &error);
        obj = json_object_get(root, "contributors");
        for(int i = 0; i< json_array_size(obj); i++){
       		 const char *strText;
       		 json_t *obj_txt;
      		 obj_txt = json_array_get(obj, i);
    	         strText = json_string_value(obj_txt);
        	 printf("found %s\n", strText);
		 if (strcmp(strText, "David")==0)
	         { 
		   count=0;
   	         }
		 else{
		 count++;
		}
	}
    }
	return count;
}


int main(){

  int  a = test();
   printf("final ans is %d\n",a);	
}
