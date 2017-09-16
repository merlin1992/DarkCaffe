#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;

	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}


//simple link for caffe_yolo

//init node
NODE *initnode(NODE *pnode, size_params data)
{
	pnode = (NODE *)malloc(sizeof(NODE));
	pnode->data = data;
	pnode->next = NULL;
	return pnode;
}

NODE *createlink_byhead(NODE*phead, size_params data){
	NODE *pnode = { 0 };
	pnode = initnode(pnode, data);
	NODE *ptmp = phead;
	if (NULL == phead){
		return pnode;
	}
	else{
		phead = pnode; 
		pnode->next = ptmp;
	}
	return phead;
}
NODE *createlink_bytail(NODE *phead, size_params data)
{
	NODE *pnode = { 0 };
	pnode = initnode(pnode, data);
	NODE *ptmp = phead;

	if (NULL == phead){
		return pnode;
	}
	else{
		while (ptmp->next != NULL){
			ptmp = ptmp->next;
		}
		ptmp->next = pnode;
	}
	return phead;
}


int linklen(NODE *phead)
{
	int len = 0;
	NODE *ptmp = phead;
	while (ptmp != NULL){
		len++;
		ptmp = ptmp->next;
	}
	return len;
}


NODE *searchnodeForLayerName(NODE *phead, char* key){

	NODE *ptmp = phead;
	if (phead == NULL) return NULL;
	while (ptmp->data.bottom_layer_name != key&&ptmp->next != NULL){
		ptmp = ptmp->next;
	}
	if (ptmp->data.bottom_layer_name == key) return ptmp;
	if (ptmp->next == NULL) return NULL;
}


int SL_Clear(NODE *phead)
{
	NODE *p = { 0 };
	NODE *q = { 0 };
	int i = 0;
	p = phead;
	while (p != NULL)
	{
		q = p->next;
		free(p);
		p = q;
	}
	phead = NULL;
	p = NULL;
	q = NULL;

	return 1;
}