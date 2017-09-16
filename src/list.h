#ifndef LIST_H
#define LIST_H

//coded by linhao for yolo_caffe 
typedef struct _node{
	size_params data;
	struct _node *next;
}NODE;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list(list *l);
void free_list_contents(list *l);

//coded by linhao for yolo_caffe
NODE *initnode(NODE *pnode, size_params data);
NODE *createlink_byhead(NODE*phead, size_params data);
NODE *createlink_bytail(NODE *phead, size_params data);
int linklen(NODE *phead);
NODE *searchnodeForLayerName(NODE *phead, char* key);
int SL_Clear(NODE *phead);

#endif
